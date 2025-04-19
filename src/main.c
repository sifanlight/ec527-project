/* quant_matmul.c – simple C version with typedef data_t
 *   (c) 2025 Sina Karimi – public‑domain / CC0
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* =====================================================
 * 1.  Choose ONE line – e.g. float or double or _Float16
 * =====================================================*/
typedef float data_t;        /* change here if you need BF16 / FP64 / etc. */

/* Handy macro for row‑major indexing of a flat array */
#define IDX(i,j,n)  ((i)*(n) + (j))

static data_t *alloc_matrix(int n);
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
                       

/* ===========================================================================*/

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <arrLen>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int n = atoi(argv[1]);
    if (n <= 0) { fputs("arrLen must be positive.\n", stderr); return 1; }

    srand((unsigned)time(NULL));

    /* 1.  allocate & fill FP32 input matrices ------------------------------*/
    data_t *A = alloc_matrix(n);
    data_t *B = alloc_matrix(n);
    init_random(A, n);
    init_random(B, n);

    /* 2.  gather row/column scale factors ---------------------------------*/
    data_t *scaleA = (data_t *)malloc(n * sizeof(data_t));
    data_t *scaleB = (data_t *)malloc(n * sizeof(data_t));
    scale_rows(A, n, scaleA);
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
    data_t *C = alloc_matrix(n);
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

static data_t *alloc_matrix(int n)
{
    data_t *m = (data_t *)malloc((size_t)n * n * sizeof(data_t));
    if (!m) { perror("malloc"); exit(EXIT_FAILURE); }
    return m;
}

static void init_random(data_t *m, int n)
{
    srand(123);
    for (int i = 0; i < n * n; ++i)
        m[i] = ((data_t)rand() / RAND_MAX) * 2.0f - 1.0f;   /* (‑1, 1) */
        // m[i] = (data_t)i;
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
