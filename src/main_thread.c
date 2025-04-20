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


// Structure to hold thread data
typedef struct {
    int thread_id;
    int n;
    const data_t** A;
    data_t** scale;
} thread_data_t;
typedef struct{
    int thread_id;
    int n;
    const data_t** A;
    data_t** scale;
    int row_or_col;
    int16_t** out;
} thread_data_t2;
#define NUM_THREADS 8

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
void *scale_rows_avx_unroll_thread_worker(void *arg);
static void scale_rows_avx_unroll_threadcall(int n, const float *A, float *scale);

static void fp32_to_int16_avx(const float *M, int n,
                              const float *row_scale,
                              const float *col_scale,
                              int row_or_col,
                              int16_t *out);

static void fp32_to_int16_avx_unroll(const float *M, int n,
                                      const float *row_scale,
                                      const float *col_scale,
                                      int row_or_col,
                                      int16_t *out);

static void *fp32_to_int16_thread_worker(void *arg);
static void fp32_to_int16_threadcall(int n, const float *A, float *scale, int row_or_col, int16_t *out);


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
    // scale_rows_avx_unroll(A, n, scaleA);
    scale_rows_avx_unroll_threadcall(n, A, scaleA);
    scale_cols(B, n, scaleB);

    /* 3.  quantise to INT16 -------------------------------------------------*/
    // int16_t *Aq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    // int16_t *Bq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    int16_t *Aq;
    int16_t *Bq;
    int ok = posix_memalign((void **)&Aq, 64, (size_t)n * n * sizeof(int16_t));
    if (ok != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    ok = posix_memalign((void **)&Bq, 64, (size_t)n * n * sizeof(int16_t));
    if (ok != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    // fp32_to_int(A, n, scaleA, NULL, 0, Aq);   /* row‑wise */
    fp32_to_int16_threadcall(n, A, scaleA, 0, Aq);   /* row‑wise */
    fp32_to_int16_avx(B, n, NULL, scaleB, 1, Bq);   /* col‑wise */

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


void *scale_rows_avx_unroll_thread_worker(void *arg)
{
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    thread_data_t *data = (thread_data_t *)arg;
    int n = data->n;
    const float *A = *(data->A);
    float *scale = *(data->scale);
    int thread_id = data->thread_id;
    int rows_per_thread = n / NUM_THREADS;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id + 1) * rows_per_thread;
    if (thread_id == NUM_THREADS - 1) {
        end_row = n; // Last thread handles any remaining rows
    }
    // printf("Thread %d processing rows %d to %d\n", thread_id, start_row, end_row);

    for (int i = start_row; i < end_row; ++i) {
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
    pthread_exit(NULL);
}

void scale_rows_avx_unroll_threadcall(int n, const float *A, float *scale)
{
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].n = n;
        thread_data[i].A = &A;
        thread_data[i].scale = &scale;
        pthread_create(&threads[i], NULL, scale_rows_avx_unroll_thread_worker, (void *)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}




static void fp32_to_int16_avx(const float *M, int n,
                                   const float *row_scale,
                                   const float *col_scale,
                                   int row_or_col,
                                   int16_t *out)
{
    const float qmax = (float)INT16_MAX;
    __m256 qmax_vec = _mm256_set1_ps(qmax);
    const int vec_len = 8;

    if (row_or_col == 0) {
        // Row-wise
        for (int i = 0; i < n; ++i) {
            __m256 scale_vec = _mm256_set1_ps(row_scale[i]);

            int j;
            for (j = 0; j <= n - vec_len; j += vec_len) {
                __m256 m_vec = _mm256_loadu_ps(&M[IDX(i, j, n)]);
                __m256 norm = _mm256_div_ps(m_vec, scale_vec);
                __m256 scaled = _mm256_mul_ps(norm, qmax_vec);
                __m256i rounded = _mm256_cvtps_epi32(scaled);

                __m128i lo = _mm256_castsi256_si128(rounded);
                __m128i hi = _mm256_extracti128_si256(rounded, 1);
                __m128i packed = _mm_packs_epi32(lo, hi);

                _mm_storeu_si128((__m128i*)&out[IDX(i, j, n)], packed);
            }

            // Tail loop
            for (; j < n; ++j) {
                float val = M[IDX(i, j, n)];
                float norm = (val / row_scale[i]) * qmax;
                out[IDX(i, j, n)] = (int16_t)lroundf(norm);
            }
            // printf("%d \n", out[IDX(i, 0, n)]);
        }

    } else {
        // Column-wise
        for (int j = 0; j < n; ++j) {
            __m256 scale_vec = _mm256_set1_ps(col_scale[j]);

            int i;
            for (i = 0; i <= n - vec_len; i += vec_len) {
                float vals[8];
                for (int k = 0; k < vec_len; ++k)
                    vals[k] = M[IDX(i + k, j, n)];

                __m256 m_vec = _mm256_loadu_ps(vals);
                __m256 norm = _mm256_div_ps(m_vec, scale_vec);
                __m256 scaled = _mm256_mul_ps(norm, qmax_vec);
                __m256i rounded = _mm256_cvtps_epi32(scaled);

                __m128i lo = _mm256_castsi256_si128(rounded);
                __m128i hi = _mm256_extracti128_si256(rounded, 1);
                __m128i packed = _mm_packs_epi32(lo, hi);

                for (int k = 0; k < vec_len; ++k)
                    out[IDX(i + k, j, n)] = ((int16_t*)&packed)[k];
            }

            // Tail loop
            for (; i < n; ++i) {
                float val = M[IDX(i, j, n)];
                float norm = (val / col_scale[j]) * qmax;
                out[IDX(i, j, n)] = (int16_t)lroundf(norm);
            }
          
        }
    }
}

static void fp32_to_int16_avx_unroll(const float *M, int n,
                                   const float *row_scale,
                                   const float *col_scale,
                                   int row_or_col,
                                   int16_t *out)
{
    const float qmax = (float)INT16_MAX;
    __m256 qmax_vec = _mm256_set1_ps(qmax);

    if (row_or_col == 0) {
        // Row-wise
        for (int i = 0; i < n; ++i) {
            __m256 scale_vec = _mm256_set1_ps(row_scale[i]);

            int j;
            for (j = 0; j <= n - 16; j += 16) {
                __m256 m_vec1 = _mm256_loadu_ps(&M[IDX(i, j, n)]);
                __m256 m_vec2 = _mm256_loadu_ps(&M[IDX(i, j + 8, n)]);
                __m256 norm1 = _mm256_div_ps(m_vec1, scale_vec);
                __m256 norm2 = _mm256_div_ps(m_vec2, scale_vec);
                __m256 scaled1 = _mm256_mul_ps(norm1, qmax_vec);
                __m256 scaled2 = _mm256_mul_ps(norm2, qmax_vec);
                __m256i rounded1 = _mm256_cvtps_epi32(scaled1);
                __m256i rounded2 = _mm256_cvtps_epi32(scaled2);

                __m128i lo1 = _mm256_castsi256_si128(rounded1);
                __m128i lo2 = _mm256_castsi256_si128(rounded2);
                __m128i hi1 = _mm256_extracti128_si256(rounded1, 1);
                __m128i hi2 = _mm256_extracti128_si256(rounded2, 1);
                __m128i packed1 = _mm_packs_epi32(lo1, hi1);
                __m128i packed2 = _mm_packs_epi32(lo2, hi2);

                _mm_storeu_si128((__m128i*)&out[IDX(i, j, n)], packed1);
                _mm_storeu_si128((__m128i*)&out[IDX(i, j + 8, n)], packed2);
            }

            // Tail loop
            for (; j < n; ++j) {
                float val = M[IDX(i, j, n)];
                float norm = (val / row_scale[i]) * qmax;
                out[IDX(i, j, n)] = (int16_t)lroundf(norm);
            }
            // printf("%d \n", out[IDX(i, 0, n)]);
        }

    } else {
        // Column-wise
        for (int j = 0; j < n; ++j) {
            __m256 scale_vec = _mm256_set1_ps(col_scale[j]);

            int i;
            for (i = 0; i <= n - 8; i += 8) {
                float vals[8];
                for (int k = 0; k < 8; ++k)
                    vals[k] = M[IDX(i + k, j, n)];

                __m256 m_vec = _mm256_loadu_ps(vals);
                __m256 norm = _mm256_div_ps(m_vec, scale_vec);
                __m256 scaled = _mm256_mul_ps(norm, qmax_vec);
                __m256i rounded = _mm256_cvtps_epi32(scaled);

                __m128i lo = _mm256_castsi256_si128(rounded);
                __m128i hi = _mm256_extracti128_si256(rounded, 1);
                __m128i packed = _mm_packs_epi32(lo, hi);

                for (int k = 0; k < 8; ++k)
                    out[IDX(i + k, j, n)] = ((int16_t*)&packed)[k];
            }

            // Tail loop
            for (; i < n; ++i) {
                float val = M[IDX(i, j, n)];
                float norm = (val / col_scale[j]) * qmax;
                out[IDX(i, j, n)] = (int16_t)lroundf(norm);
            }
          
        }
    }
}


void *fp32_to_int16_thread_worker(void *arg)
{
    thread_data_t2 *data = (thread_data_t2 *)arg;
    int n = data->n;
    int row_or_col = data->row_or_col;
    const float *M = *(data->A);
    float *scale = *(data->scale);
    int thread_id = data->thread_id;
    int rows_per_thread = n / NUM_THREADS;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id + 1) * rows_per_thread;
    int16_t *out = *(data->out);
    if (thread_id == NUM_THREADS - 1) {
        end_row = n; // Last thread handles any remaining rows
    }
    printf("Thread %d processing rows %d to %d\n", thread_id, start_row, end_row);

    const float qmax = (float)INT16_MAX;
    __m256 qmax_vec = _mm256_set1_ps(qmax);

    if (row_or_col == 0) {
        // Row-wise
        for (int i = start_row; i < end_row; ++i) {
            __m256 scale_vec = _mm256_set1_ps(scale[i]);

            int j;
            for (j = 0; j <= n - 16; j += 16) {
                __m256 m_vec1 = _mm256_loadu_ps(&M[IDX(i, j, n)]);
                __m256 m_vec2 = _mm256_loadu_ps(&M[IDX(i, j + 8, n)]);
                __m256 norm1 = _mm256_div_ps(m_vec1, scale_vec);
                __m256 norm2 = _mm256_div_ps(m_vec2, scale_vec);
                __m256 scaled1 = _mm256_mul_ps(norm1, qmax_vec);
                __m256 scaled2 = _mm256_mul_ps(norm2, qmax_vec);
                __m256i rounded1 = _mm256_cvtps_epi32(scaled1);
                __m256i rounded2 = _mm256_cvtps_epi32(scaled2);

                __m128i lo1 = _mm256_castsi256_si128(rounded1);
                __m128i lo2 = _mm256_castsi256_si128(rounded2);
                __m128i hi1 = _mm256_extracti128_si256(rounded1, 1);
                __m128i hi2 = _mm256_extracti128_si256(rounded2, 1);
                __m128i packed1 = _mm_packs_epi32(lo1, hi1);
                __m128i packed2 = _mm_packs_epi32(lo2, hi2);

                _mm_storeu_si128((__m128i*)&out[IDX(i, j, n)], packed1);
                _mm_storeu_si128((__m128i*)&out[IDX(i, j + 8, n)], packed2);
            }

            // Tail loop
            for (; j < n; ++j) {
                float val = M[IDX(i, j, n)];
                float norm = (val / scale[i]) * qmax;
                out[IDX(i, j, n)] = (int16_t)lroundf(norm);
            }
            // printf("%d \n", out[IDX(i, 0, n)]);
        }

    } else {
        // Column-wise
        for (int j = start_row; j < end_row; ++j) {
            __m256 scale_vec = _mm256_set1_ps(scale[j]);

            int i;
            for (i = 0; i <= n - 8; i += 8) {
                float vals[8];
                for (int k = 0; k < 8; ++k)
                    vals[k] = M[IDX(i + k, j, n)];

                __m256 m_vec = _mm256_loadu_ps(vals);
                __m256 norm = _mm256_div_ps(m_vec, scale_vec);
                __m256 scaled = _mm256_mul_ps(norm, qmax_vec);
                __m256i rounded = _mm256_cvtps_epi32(scaled);

                __m128i lo = _mm256_castsi256_si128(rounded);
                __m128i hi = _mm256_extracti128_si256(rounded, 1);
                __m128i packed = _mm_packs_epi32(lo, hi);

                for (int k = 0; k < 8; ++k)
                    out[IDX(i + k, j, n)] = ((int16_t*)&packed)[k];
            }

            // Tail loop
            for (; i < n; ++i) {
                float val = M[IDX(i, j, n)];
                float norm = (val / scale[j]) * qmax;
                out[IDX(i, j, n)] = (int16_t)lroundf(norm);
            }
          
        }
    }

    pthread_exit(NULL);
}


static void fp32_to_int16_threadcall(int n, const float *A, float *scale, int row_or_col, int16_t *out)
{
    pthread_t threads[NUM_THREADS];
    thread_data_t2 thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].n = n;
        thread_data[i].A = &A;
        thread_data[i].scale = &scale;
        thread_data[i].row_or_col = row_or_col;
        thread_data[i].out = &out;
        pthread_create(&threads[i], NULL, fp32_to_int16_thread_worker, (void *)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}