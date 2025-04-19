/* quant_matmul.c – simple C version with typedef data_t
 *   (c) 2025 Sina Karimi – public‑domain / CC0
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <pthread.h>

/* =====================================================
 * 1.  Choose ONE line – e.g. float or double or _Float16
 * =====================================================*/
typedef float data_t;        /* change here if you need BF16 / FP64 / etc. */

/* Handy macro for row‑major indexing of a flat array */
#define IDX(i,j,n)  ((i)*(n) + (j))

static void alloc_matrix(int n, data_t** m);
static void init_random(data_t *m, int n); 
static void scale_rows(const data_t *A, int n, data_t *scale);
static void scale_cols(const data_t *B, int n, data_t *scale);
static void fp32_to_int16(const data_t *M, int n,
                          const data_t *row_scale,
                          const data_t *col_scale,
                          int row_or_col,
                          int16_t *out);
static void matmul_int16(const int16_t *A, const int16_t *B,
                         int n, int64_t *C);
static void dequantise(const int64_t *Cq, int n,
                       const data_t *scaleA, const data_t *scaleB,
                       data_t *C);
                       

static inline float hmax256(__m256 v);
static void scale_rows_avx(const float *A, int n, float *scale);
static void scale_rows_avx_unroll(const float *A, int n, float *scale);

/* ===========================================================================*/

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <arrLen>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int n = atoi(argv[1]);
    if (n <= 0) { fputs("arrLen must be positive.\n", stderr); return 1; }

    // srand((unsigned)time(NULL));

    /* 1.  allocate & fill FP32 input matrices ------------------------------*/
    // data_t *A = alloc_matrix(n);
    // data_t *B = alloc_matrix(n);
    data_t *A;
    data_t *B;
    alloc_matrix(n, &A);
    alloc_matrix(n, &B);
    init_random(A, n);
    init_random(B, n);

    /* 2.  gather row/column scale factors ---------------------------------*/
    data_t *scaleA = (data_t *)malloc(n * sizeof(data_t));
    data_t *scaleB = (data_t *)malloc(n * sizeof(data_t));
    // scale_rows(A, n, scaleA);
    scale_rows_avx_unroll(A, n, scaleA);
    scale_cols(B, n, scaleB);

    /* 3.  quantise to INT16 -------------------------------------------------*/
    int16_t *Aq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    int16_t *Bq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    fp32_to_int16(A, n, scaleA, NULL, 0, Aq);   /* row‑wise */
    fp32_to_int16(B, n, NULL, scaleB, 1, Bq);   /* col‑wise */

    /* 4.  INT16 matmul with INT64 output -----------------------------------*/
    int64_t *Cq = (int64_t *)malloc((size_t)n * n * sizeof(int64_t));
    matmul_int16(Aq, Bq, n, Cq);

    /* 5.  de‑quantise back to FP32 -----------------------------------------*/
    data_t *C;
    alloc_matrix(n, &C);
    dequantise(Cq, n, scaleA, scaleB, C);


    /* 6.  print result (for debugging) ------------------------------------*/
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%8.4f ", C[IDX(i, j, n)]);
        putchar('\n');
    }
    /* 6.  cleanup ----------------------------------------------------------*/
    free(A);  free(B);  free(C);
    free(Aq); free(Bq); free(Cq);
    free(scaleA); free(scaleB);

    return 0;
}



/* ---------- basic utilities ------------------------------------------------*/

static void alloc_matrix(int n, data_t** m)
{
    int ok = posix_memalign((void **)m, 64, (size_t)n * n * sizeof(data_t));
    if (ok != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
}


static void init_random(data_t *m, int n)
{
    srand(123);
    for (int i = 0; i < n * n; ++i)
        m[i] = ((data_t)rand() / RAND_MAX) * 2.0f - 1.0f;   /* (‑1, 1) */
        // m[i] = (data_t)1;
}

/* ---------- scaling factors -------------------------------------------------*/

static void scale_rows(const data_t *A, int n, data_t *scale)
{
    for (int i = 0; i < n; ++i) {
        data_t maxv = 0.0f;
        for (int j = 0; j < n; ++j) {
            data_t v = fabsf(A[IDX(i, j, n)]);
            if (v > maxv) maxv = v;
        }
        scale[i] = maxv;
    }
}

static void scale_cols(const data_t *B, int n, data_t *scale)
{
    for (int j = 0; j < n; ++j) {
        data_t maxv = 0.0f;
        for (int i = 0; i < n; ++i) {
            data_t v = fabsf(B[IDX(i, j, n)]);
            if (v > maxv) maxv = v;
        }
        scale[j] = maxv;
    }
}

/* row_or_col = 0  →  row‑wise (matrix A);  1 → column‑wise (matrix B) */
static void fp32_to_int16(const data_t *M, int n,
                          const data_t *row_scale,
                          const data_t *col_scale,
                          int row_or_col,
                          int16_t *out)
{
    const int16_t qmax = INT16_MAX;

    if (row_or_col == 0) {          /* row‑wise */
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                out[IDX(i, j, n)] =
                    (int16_t)lroundf((M[IDX(i, j, n)] / row_scale[i]) * qmax);

    } else {                        /* column‑wise */
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                out[IDX(i, j, n)] =
                    (int16_t)lroundf((M[IDX(i, j, n)] / col_scale[j]) * qmax);
    }
}

/* ---------- INT16 GEMM with INT64 accumulation -----------------------------*/

static void matmul_int16(const int16_t *A, const int16_t *B,
                         int n, int64_t *C)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (int k = 0; k < n; ++k)
                acc += (int64_t)A[IDX(i, k, n)] * (int64_t)B[IDX(k, j, n)];
            C[IDX(i, j, n)] = acc;
        }
    }
}

/* ---------- de‑quantisation -------------------------------------------------*/

static void dequantise(const int64_t *Cq, int n,
                       const data_t *scaleA, const data_t *scaleB,
                       data_t *C)
{
    const int16_t qmax = INT16_MAX;
    const data_t   inv = 1.0f / ((data_t)qmax * (data_t)qmax);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[IDX(i, j, n)] = (data_t)Cq[IDX(i, j, n)] *
                              scaleA[i] * scaleB[j] * inv;
}


/* ===========================================================================*/
static inline float hmax256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1); // high 128
    __m128 lo = _mm256_castps256_ps128(v);   // low 128
    __m128 max128 = _mm_max_ps(lo, hi);      // max across halves

    // Reduce within 128-bit lane
    __m128 shuf = _mm_movehdup_ps(max128);   // (1,1,3,3)
    __m128 max2 = _mm_max_ps(max128, shuf);
    shuf = _mm_movehl_ps(shuf, max2);
    __m128 max4 = _mm_max_ss(max2, shuf);
    return _mm_cvtss_f32(max4);
}

void scale_rows_avx(const float *A, int n, float *scale)
{
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (int i = 0; i < n; ++i) {
        const float *row = &A[i * n];
        __m256 maxv = _mm256_setzero_ps();

        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m256 v = _mm256_loadu_ps(&row[j]);
            v = _mm256_and_ps(v, signmask);        // fabs
            maxv = _mm256_max_ps(maxv, v);
        }

        float max_scalar = hmax256(maxv);          // reduce vector to scalar

        // Handle tail elements (n not divisible by 8)
        for (; j < n; ++j) {
            float v = fabsf(row[j]);
            if (v > max_scalar) max_scalar = v;
        }

        scale[i] = max_scalar;
    }
}

void scale_rows_avx_unroll(const float *A, int n, float *scale)
{
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (int i = 0; i < n; ++i) {
        const float *row = &A[i * n];
        __m256 maxv1 = _mm256_setzero_ps();
        __m256 maxv2 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= n - 16; j += 16) {
            __m256 v1 = _mm256_loadu_ps(&row[j]);
            __m256 v2 = _mm256_loadu_ps(&row[j + 8]);
            v1 = _mm256_and_ps(v1, signmask);        // fabs
            v2 = _mm256_and_ps(v2, signmask);        // fabs
            maxv1 = _mm256_max_ps(maxv1, v1);
            maxv2 = _mm256_max_ps(maxv2, v2);
        }

        float max_scalar1 = hmax256(maxv1);          // reduce vector to scalar
        float max_scalar2 = hmax256(maxv2);          // reduce vector to scalar
        float max_scalar = max_scalar1 > max_scalar2 ? max_scalar1 : max_scalar2;

        // Handle tail elements (n not divisible by 8)
        for (; j < n; ++j) {
            float v = fabsf(row[j]);
            if (v > max_scalar) max_scalar = v;
        }

        scale[i] = max_scalar;
    }
}