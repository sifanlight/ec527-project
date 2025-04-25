/* quant_matmul.c – simple C version with typedef data_t
 *   (c) 2025 Sina Karimi – public‑domain / CC0
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
// #include <iostream>
// #include <fstream>
// #include <cmath> // For std::round
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <pthread.h>
#include <limits.h>
#include <assert.h>
// #include <iostream>
#include <math.h>
// #include <iostream>
// #include <fstream>
#include <string.h>
#include <time.h>
#include <cmath>                            // For std::round
#include <boost/multiprecision/cpp_int.hpp> // For int128_t
#include <smmintrin.h>
using namespace boost::multiprecision;

/* =====================================================
 * 1.  Choose ONE line – e.g. float or double or _Float16
 * =====================================================*/
typedef float data_t; /* change here if you need BF16 / FP64 / etc. */
typedef struct
{
    int n;
    int thread_id;
    const int16_t **mat;
    const uint8_t **moduli_set;
    uint8_t **res_3D_mat;
} thread_RNS_forward_data_t;

typedef struct {
    const uint8_t *res_MM_3D_mat;
    int n;
    int m;
    int64_t *rns_to_int_mat;
    int thread_id;
} thread_CRT_MMM_data_t;

#define NUM_THREADS 32

/* Handy macro for row‑major indexing of a flat array */
#define IDX(i, j, n) ((i) * (n) + (j))
#define IDX_3D(i, j, k, n) ((i) * (n) + (j) + k * (n * n))
static void alloc_matrix(int n, data_t **m);
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
static void matmul_float(const float *A, const float *B,
                         int n, float *C);
static void dequantise(const int64_t *Cq, int n,
                       const data_t *scaleA, const data_t *scaleB,
                       data_t *C);

int128_t gcdExtended(int128_t a, int128_t b, int128_t *x, int128_t *y);
int128_t modInverse(int128_t A, int128_t M);
uint8_t floored_mod_forward(int16_t a, uint8_t b);
static void forward_RNS(const int16_t *mat, int n,        // int n is the size of the matrix
                        const uint8_t *moduli_set, int m, // int m is the size of the moduli set
                        uint8_t *res_3D_mat);
uint8_t floored_mod_MM(uint32_t a, uint8_t b);
static void RNS_MMM(const uint8_t *mat_A_3D, const uint8_t *mat_B_3D,
                    int n, int m, const uint8_t *moduli_set,
                    uint8_t *res_3D_mat);
int128_t floored_mod_CRT(int128_t a, int128_t b);
static void CRT_MMM(const uint8_t *res_MM_3D_mat, int n, int m,
                    int64_t *rns_to_int_mat);

static void forward_RNS_avx(const int16_t *mat, int n,
                            const uint8_t *moduli_set, int m,
                            uint8_t *res_3D_mat);

static void RNS_MMM_vec(const uint8_t *mat_A_3D, const uint8_t *mat_B_3D,
                        int n, int m, const uint8_t *moduli_set, uint8_t *res_3D_mat);
static void forward_RNS_avx_unroll_threadcall(const int16_t *mat, int n,
                                              const uint8_t *moduli_set, int m,
                                              uint8_t *res_3D_mat);
static void forward_RNS_avx_unroll(const int16_t *mat, int n,
                                   const uint8_t *moduli_set, int m,
                                   uint8_t *res_3D_mat);
void *forward_RNS_avx_unroll_thread_worker(void *arg);

void *CRT_MMM_worker(void *arg);
static void CRT_MMM_threadcall(const uint8_t *res_MM_3D_mat, int n, int m,
                    int64_t *rns_to_int_mat);

/* ===========================================================================*/

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int n = atoi(argv[1]);
    if (n <= 0)
    {
        fputs("matrix_size must be positive.\n", stderr);
        return 1;
    }

    srand((unsigned)time(NULL));

    /* Moduli set */
    // const uint8_t moduli_set[8] = {254, 253, 251, 249, 247, 245, 241, 239};//, 233};
    uint8_t *moduli_set;
    int ok = posix_memalign((void **)&moduli_set, 32, 8 * sizeof(uint8_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    moduli_set[0] = 254;
    moduli_set[1] = 253;
    moduli_set[2] = 251;
    moduli_set[3] = 249;
    moduli_set[4] = 247;
    moduli_set[5] = 245;
    moduli_set[6] = 241;
    moduli_set[7] = 239;
    const int m = 8; // 9 moduli

    /* 1. allocate & fill FP32 input matrices ------------------------------ */
    // data_t *A = alloc_matrix(n);
    // data_t *B = alloc_matrix(n);
    // data_t *C_res = alloc_matrix(n);
    data_t *A;
    data_t *B;
    data_t *C_res;
    alloc_matrix(n, &A);
    alloc_matrix(n, &B);
    alloc_matrix(n, &C_res);
    init_random(A, n);
    init_random(B, n);

    matmul_float(A, B, n, C_res);
    /* 8. print result ------------------------------------------------------ */
    // printf("\n input result:\n");
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j)
    //         printf("%8.4f ", C_res[IDX(i, j, n)]);
    //     putchar('\n');
    // }

    /* 2. scaling factors -------------------------------------------------- */
    // data_t *scaleA = (data_t *)malloc(n * sizeof(data_t));
    // data_t *scaleB = (data_t *)malloc(n * sizeof(data_t));
    data_t *scaleA;
    data_t *scaleB;
    // int ok;
    ok = posix_memalign((void **)&scaleA, 32, n * sizeof(data_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    ok = posix_memalign((void **)&scaleB, 32, n * sizeof(data_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    scale_rows(A, n, scaleA);
    scale_cols(B, n, scaleB);

    /* 3. quantise to INT16 ------------------------------------------------ */
    // int16_t *Aq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    // int16_t *Bq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    int16_t *Aq;
    int16_t *Bq;
    ok = posix_memalign((void **)&Aq, 64, (size_t)n * n * sizeof(int16_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    ok = posix_memalign((void **)&Bq, 64, (size_t)n * n * sizeof(int16_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    fp32_to_int16(A, n, scaleA, NULL, 0, Aq); /* row-wise */
    fp32_to_int16(B, n, NULL, scaleB, 1, Bq); /* col-wise */

    /* 4. forward RNS conversion ------------------------------------------- */
    // uint8_t *Aq_3D = (uint8_t *)malloc((size_t)n * n * m * sizeof(uint8_t));
    // uint8_t *Bq_3D = (uint8_t *)malloc((size_t)n * n * m * sizeof(uint8_t));
    uint8_t *Aq_3D;
    uint8_t *Bq_3D;
    ok = posix_memalign((void **)&Aq_3D, 64, (size_t)n * n * m * sizeof(uint8_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    ok = posix_memalign((void **)&Bq_3D, 64, (size_t)n * n * m * sizeof(uint8_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    // forward_RNS(Aq, n, moduli_set, m, Aq_3D);
    // forward_RNS(Bq, n, moduli_set, m, Bq_3D);
    forward_RNS_avx_unroll_threadcall(Aq, n, moduli_set, m, Aq_3D);
    forward_RNS_avx_unroll_threadcall(Bq, n, moduli_set, m, Bq_3D);

    /* 5. RNS MMM ----------------------------------------------------------- */
    // uint8_t *Cq_3D = (uint8_t *)malloc((size_t)n * n * m * sizeof(uint8_t));
    uint8_t *Cq_3D;
    ok = posix_memalign((void **)&Cq_3D, 64, (size_t)n * n * m * sizeof(uint8_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    RNS_MMM_vec(Aq_3D, Bq_3D, n, m, moduli_set, Cq_3D);

    /* 6. CRT reconstruction ------------------------------------------------ */
    // int64_t *Cq = (int64_t *)malloc((size_t)n * n * sizeof(int64_t));
    int64_t *Cq;
    ok = posix_memalign((void **)&Cq, 64, (size_t)n * n * sizeof(int64_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    CRT_MMM_threadcall(Cq_3D, n, m, Cq);

    /* 7. Dequantise -------------------------------------------------------- */
    // data_t *C = alloc_matrix(n);
    data_t *C;
    ok = posix_memalign((void **)&C, 32, n * n * sizeof(data_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    dequantise(Cq, n, scaleA, scaleB, C);

    /* 8. print result ------------------------------------------------------ */
    // printf("\nResult matrix (after RNS MatMul + CRT + Dequantization):\n");
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j)
    //         printf("%8.4f ", C[IDX(i, j, n)]);
    //     putchar('\n');
    // }
    /* comparing the result of RNS and normal*/

    printf("\nComparing the result of RNS and normal multiplication:\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (fabs(C[IDX(i, j, n)] - C_res[IDX(i, j, n)]) > 1e-2)
            {
                printf("Mismatch at (%d, %d): RNS = %8.4f, Normal = %8.4f\n", i, j, C[IDX(i, j, n)], C_res[IDX(i, j, n)]);
            }
        }
    }

    /* 9. cleanup ----------------------------------------------------------- */
    free(A);
    free(B);
    free(C);
    free(Aq);
    free(Bq);
    free(Cq);
    free(Aq_3D);
    free(Bq_3D);
    free(Cq_3D);
    free(scaleA);
    free(scaleB);

    return 0;
}

/* ---------- basic utilities ------------------------------------------------*/

static void alloc_matrix(int n, data_t **m)
{
    int ok = posix_memalign((void **)m, 64, (size_t)n * n * sizeof(data_t));
    if (ok != 0)
    {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
}

static void init_random(data_t *m, int n)
{
    srand(123);
    for (int i = 0; i < n * n; ++i)
        m[i] = ((data_t)rand() / RAND_MAX) * 2.0f - 1.0f; /* (‑1, 1) */
                                                          // m[i] = (data_t)i;
}

/* ---------- scaling factors -------------------------------------------------*/

static void scale_rows(const data_t *A, int n, data_t *scale)
{
    for (int i = 0; i < n; ++i)
    {
        data_t maxv = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            data_t v = fabsf(A[IDX(i, j, n)]);
            if (v > maxv)
                maxv = v;
        }
        scale[i] = maxv;
    }
}

static void scale_cols(const data_t *B, int n, data_t *scale)
{
    for (int j = 0; j < n; ++j)
    {
        data_t maxv = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            data_t v = fabsf(B[IDX(i, j, n)]);
            if (v > maxv)
                maxv = v;
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

    if (row_or_col == 0)
    { /* row‑wise */
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                out[IDX(i, j, n)] =
                    (int16_t)lroundf((M[IDX(i, j, n)] / row_scale[i]) * qmax);
    }
    else
    { /* column‑wise */
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
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int64_t acc = 0;
            for (int k = 0; k < n; ++k)
                acc += (int64_t)A[IDX(i, k, n)] * (int64_t)B[IDX(k, j, n)];
            C[IDX(i, j, n)] = acc;
        }
    }
}

static void matmul_float(const float *A, const float *B,
                         int n, float *C)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float acc = 0;
            for (int k = 0; k < n; ++k)
                acc += (float)A[IDX(i, k, n)] * (float)B[IDX(k, j, n)];
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
    const data_t inv = 1.0f / ((data_t)qmax * (data_t)qmax);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[IDX(i, j, n)] = (data_t)Cq[IDX(i, j, n)] *
                              scaleA[i] * scaleB[j] * inv;
}

/*--------Utility Functions for pre-computed constants----------*/
int128_t gcdExtended(int128_t a, int128_t b, int128_t *x, int128_t *y)
{

    // Base Case
    if (a == 0)
    {
        *x = 0, *y = 1;
        return b;
    }

    // To store results of recursive call
    int128_t x1, y1;
    int128_t gcd = gcdExtended(b % a, a, &x1, &y1);

    // Update x and y using results of recursive
    // call
    *x = y1 - (b / a) * x1;
    *y = x1;

    return gcd;
}
int128_t modInverse(int128_t A, int128_t M)
{

    for (int128_t X = 1; X < M; X++)
        if ((A * X) % M == 1)
            return X;
}

/*---------------------------------------------------------------*/
uint8_t inline floored_mod_forward(int16_t a, uint8_t b)
{ // to be used in forward RNS
    return static_cast<uint8_t>(((a % static_cast<int16_t>(b)) + static_cast<int16_t>(b)) % static_cast<int16_t>(b));
}

static void forward_RNS(const int16_t *mat, int n,        // int n is the size of the matrix
                        const uint8_t *moduli_set, int m, // int m is the size of the moduli set
                        uint8_t *res_3D_mat)
{
    /*
    Calculates residues for each element of the matrix.
    Returns a 3D matrix each element is a 1x8 vector containing RNS representation of the corresponding element.
    */
    // int8_t *res_vector = (int8_t *)malloc(n * sizeof(int8_t));
    //  First calculate residues using floored modulus
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < m; k++)
            {
                res_3D_mat[IDX_3D(i, j, k, n)] = floored_mod_forward((mat[IDX(i, j, n)]), moduli_set[k]);
            }
        }
    }
    // res_3D_mat =[8xNxN] --> res_3D_mat[0*(n*n) + 0*n + 0] ,
}
uint8_t inline floored_mod_MM(uint32_t a, uint8_t b)
{ // to be used in forward RNS
    return static_cast<uint8_t>(((a % static_cast<uint32_t>(b)) + static_cast<uint32_t>(b)) % static_cast<uint32_t>(b));
}
static void RNS_MMM(const uint8_t *mat_A_3D, const uint8_t *mat_B_3D, int n, int m, const uint8_t *moduli_set, uint8_t *res_3D_mat) // nxnxm matrices where m=size of the moduli set
                                                                                                                                    //                      n=size of the square matrices
{
    /*
    Performs matrix matrix multiplication in RNS domain, 3D matrix multiplication WITH mod operation.
    Each "layer" corresponding to a modulus is independent.
    For TC, we need to discard mod thingie mingie here.

    */
    for (int k = 0; k < m; ++k)
    { // For each modulus
        for (int i = 0; i < n; ++i)
        { // For each row of A
            for (int j = 0; j < n; ++j)
            { // For each column of B
                uint32_t sum = 0;
                for (int l = 0; l < n; ++l)
                { // Sum over row of A and column of B
                    uint32_t a = mat_A_3D[IDX_3D(i, l, k, n)];
                    uint32_t b = mat_B_3D[IDX_3D(l, j, k, n)];
                    sum += a * b; // mod thingie mingie here
                }
                res_3D_mat[IDX_3D(i, j, k, n)] = floored_mod_MM(sum, moduli_set[k]); // You can remove mod if needed as res_3D_mat[IDX_3D(i, j, k, n)] = sum;
            }
        }
    }
}

int128_t floored_mod_CRT(int128_t a, int128_t b)
{
    return static_cast<int128_t>(((a % b) + b) % b);
}

/*
m={254,253,251,249,247,245,241,239,233} i=0,...,8
We need to pre-calculate intermediate values such that  M=prod(m_i), M_i=M\m_i without overflow, i.e. we need 72-bit for M AND 64-bit for M_i's
I am using int128_t for all for now

res_MM_3D_mat: (n x n x m) flattened 3D matrix, where each element is 8-bit residue
rns_to_int_mat: output (n x n) flattened 2D matrix, where each element is 64-bit normal integer

*/

// Moduli array (your m_i's)

// Precomputed M and M_i values--- declare as constants once calcualting them
// int128_t M = int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249) * int128_t(247) * int128_t(245) * int128_t(241) * int128_t(239) ;//* int128_t(233);
// for (int k = 0; k < 8; ++k) {
//     M_i[k] = 1;
//     for (int l = 0; l < 8; ++l) {
//         if (l != k) {
//             M_i[k] *= int128_t(moduli_set[l]);
//         }
//     }
// }
// for (int i = 0; i < 8; i++) {
//     std::cout << "M_i[" << i << "] = " << M_i[i] << std::endl;
// }
// for (int k = 0; k < 8; ++k) {
//     inv_M_i[k] = modInverse(M_i[k], int128_t(moduli_set[k]));
// }
// for (int i = 0; i < 8; i++) {
//     std::cout << "inv_M_i[" << i << "] = " << inv_M_i[i] << std::endl;
// }
// std::cout << "M = " << M << std::endl;
/* The value for each of the M_i and inv_M_I is here:
M = 13999266705215721930
M_i[0] = 55115223248880795
M_i[1] = 55333069981089810
M_i[2] = 55773970937114430
M_i[3] = 56221954639420570
M_i[4] = 56677193138525190
M_i[5] = 57139864102921314
M_i[6] = 58088243590106730
M_i[7] = 58574337678726870
inv_M_i[0] = 191
inv_M_i[1] = 84
inv_M_i[2] = 177
inv_M_i[3] = 196
inv_M_i[4] = 225
inv_M_i[5] = 64
inv_M_i[6] = 227
inv_M_i[7] = 74
*/

/* i will just declare them as constants after calcualting them
int128_t M= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_0= int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_1= int128_t(254) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_2= int128_t(254) * int128_t(253) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_3= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(247)* int128_t(245)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_4= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(245)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_5= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(241)* int128_t(239)* int128_t(233);
int128_t M_6= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(239)* int128_t(233);
int128_t M_7= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(241)* int128_t(233);
int128_t M_8= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247) * int128_t(245) * int128_t(241) * int128_t(239);

int128_t M_i[9]={M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8}
*/
// Precompute modular inverses mod m_i
// Now for each (i,j) position do CRT reconstruction
static void CRT_MMM(const uint8_t *res_MM_3D_mat, int n, int m,
                    int64_t *rns_to_int_mat)
{

    const int moduli_set[8] = {254, 253, 251, 249, 247, 245, 241, 239}; //, 233};
    int128_t M_i[8];
    int128_t inv_M_i[8];
    int128_t M = int128_t("13999266705215721930");
    int128_t lambda = (M - 1) / 2;
    M_i[0] = int128_t("55115223248880795");
    M_i[1] = int128_t("55333069981089810");
    M_i[2] = int128_t("55773970937114430");
    M_i[3] = int128_t("56221954639420570");
    M_i[4] = int128_t("56677193138525190");
    M_i[5] = int128_t("57139864102921314");
    M_i[6] = int128_t("58088243590106730");
    M_i[7] = int128_t("58574337678726870");
    inv_M_i[0] = int128_t(191);
    inv_M_i[1] = int128_t(84);
    inv_M_i[2] = int128_t(177);
    inv_M_i[3] = int128_t(196);
    inv_M_i[4] = int128_t(225);
    inv_M_i[5] = int128_t(64);
    inv_M_i[6] = int128_t(227);
    inv_M_i[7] = int128_t(74);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int128_t sum = 0;
            for (int k = 0; k < m; ++k)
            {
                uint8_t res_ijk = res_MM_3D_mat[IDX_3D(i, j, k, n)];
                int128_t term = int128_t(res_ijk) * M_i[k] * inv_M_i[k];
                sum += term;
            }
            // Modulo M to wrap around
            int128_t sum_t = floored_mod_CRT(sum, M);
            if (sum_t <= lambda)
            {
                rns_to_int_mat[i * n + j] = static_cast<int64_t>(sum_t);
            }
            else
            {
                rns_to_int_mat[i * n + j] = static_cast<int64_t>(sum_t - M);
            }

            // Save final result as 64-bit integer (should fit safely)
            // rns_to_int_mat[i*n + j] = sum_t;
        }
    }
}

// // Vectorized wrapper for floored_mod_forward using scalar fallback
// inline __m128i floored_mod_forward_vec(__m128i a, __m128i b) {
//     alignas(16) int16_t a_arr[8], b_arr[8], res_arr[8];
//     _mm_store_si128((__m128i*)a_arr, a);
//     _mm_store_si128((__m128i*)b_arr, b);

//     for (int i = 0; i < 8; ++i) {
//         int16_t aa = a_arr[i];
//         int16_t bb = b_arr[i];
//         // Safe floored modulus (always returns positive mod)
//         int16_t mod = ((aa % bb) + bb) % bb;
//         res_arr[i] = mod;
//     }

//     return _mm_load_si128((__m128i*)res_arr);
// }

/// floor-positive modulus element-wise on eight signed 16-bit lanes
/// r = ((a % b) + b) % b  , with  0 ≤ r < b   (b must be >0, non-zero)
inline __m128i floored_mod_forward_vec(__m128i a16, __m128i b16)
{
    // printf("I'm here\n");
    // 1. Unpack 8×i16 -> two 4×i32 blocks (sign-extended)
    __m128i a_lo32 = _mm_cvtepi16_epi32(a16); // a[0..3]
    __m128i b_lo32 = _mm_cvtepi16_epi32(b16);
    __m128i a_hi32 = _mm_cvtepi16_epi32(_mm_srli_si128(a16, 8)); // a[4..7]
    __m128i b_hi32 = _mm_cvtepi16_epi32(_mm_srli_si128(b16, 8));

    // 2. Convert to float and compute floor-quotient q = ⎣a / b⎦
    __m128 q_lo_f = _mm_div_ps(_mm_cvtepi32_ps(a_lo32),
                               _mm_cvtepi32_ps(b_lo32));
    __m128 q_hi_f = _mm_div_ps(_mm_cvtepi32_ps(a_hi32),
                               _mm_cvtepi32_ps(b_hi32));

    q_lo_f = _mm_round_ps(q_lo_f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    q_hi_f = _mm_round_ps(q_hi_f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    __m128i q_lo32 = _mm_cvttps_epi32(q_lo_f);
    __m128i q_hi32 = _mm_cvttps_epi32(q_hi_f);

    // 3. r = a – q*b   (now guaranteed 0 ≤ r < b)
    __m128i r_lo32 = _mm_sub_epi32(a_lo32, _mm_mullo_epi32(q_lo32, b_lo32));
    __m128i r_hi32 = _mm_sub_epi32(a_hi32, _mm_mullo_epi32(q_hi32, b_hi32));

    // 4. Pack back to 8×i16 and return
    return _mm_packs_epi32(r_lo32, r_hi32);
}

static void forward_RNS_avx(const int16_t *mat, int n,
                            const uint8_t *moduli_set, int m,
                            uint8_t *res_3D_mat)
{
    assert(m == 8); // this implementation only supports 8 moduli

    // Load all 8 moduli as a __m128i vector
    __m128i moduli = _mm_set_epi16(
        moduli_set[7], moduli_set[6], moduli_set[5], moduli_set[4],
        moduli_set[3], moduli_set[2], moduli_set[1], moduli_set[0]);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int16_t val = mat[IDX(i, j, n)];
            __m128i a_vec = _mm_set1_epi16(val);

            // Vectorized floored mod
            __m128i res_vec = floored_mod_forward_vec(a_vec, moduli);

            // Store to 3D matrix (res_3D_mat[k*(n*n) + i*n + j])
            // Since m == 8, k in [0,7]
            alignas(16) uint16_t tmp[8];
            _mm_store_si128((__m128i *)tmp, res_vec);

            for (int k = 0; k < 8; ++k)
            {
                res_3D_mat[IDX_3D(i, j, k, n)] = static_cast<uint8_t>(tmp[k]);
            }
        }
    }
}

static void forward_RNS_avx_unroll(const int16_t *mat, int n,
                                   const uint8_t *moduli_set, int m,
                                   uint8_t *res_3D_mat)
{
    assert(m == 8); // this implementation only supports 8 moduli

    // Load all 8 moduli as a __m128i vector
    __m128i moduli = _mm_set_epi16(
        moduli_set[7], moduli_set[6], moduli_set[5], moduli_set[4],
        moduli_set[3], moduli_set[2], moduli_set[1], moduli_set[0]);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; j = j + 2)
        {
            int16_t val = mat[IDX(i, j, n)];
            int16_t val2 = mat[IDX(i, j + 1, n)];
            __m128i a_vec = _mm_set1_epi16(val);
            __m128i a_vec2 = _mm_set1_epi16(val2);

            // Vectorized floored mod
            __m128i res_vec = floored_mod_forward_vec(a_vec, moduli);
            __m128i res_vec2 = floored_mod_forward_vec(a_vec2, moduli);

            // Store to 3D matrix (res_3D_mat[k*(n*n) + i*n + j])
            // Since m == 8, k in [0,7]
            alignas(16) uint16_t tmp[8];

            _mm_store_si128((__m128i *)tmp, res_vec);

            for (int k = 0; k < 8; ++k)
            {
                res_3D_mat[IDX_3D(i, j, k, n)] = static_cast<uint8_t>(tmp[k]);
            }
            _mm_store_si128((__m128i *)tmp, res_vec2);
            for (int k = 0; k < 8; ++k)
            {
                res_3D_mat[IDX_3D(i, j + 1, k, n)] = static_cast<uint8_t>(tmp[k]);
            }
        }
    }
}

void *forward_RNS_avx_unroll_worker(void *arg)
{
    // Unpack arguments
    thread_RNS_forward_data_t *data = (thread_RNS_forward_data_t *)arg;
    int n = data->n;
    int m = 8;
    int thread_id = data->thread_id;
    const int16_t *mat = *(data->mat);
    const uint8_t *moduli_set = *(data->moduli_set);
    uint8_t *res_3D_mat = *(data->res_3D_mat);
    int rows_per_thread = n / NUM_THREADS;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id + 1) * rows_per_thread;
    if (thread_id == NUM_THREADS - 1)
    {
        end_row = n; // Last thread takes the remaining rows
    }
    // printf("Thread %d processing rows %d to %d\n", thread_id, start_row, end_row);
    // assert(m == 8); // this implementation only supports 8 moduli

    // Load all 8 moduli as a __m128i vector
    __m128i moduli = _mm_set_epi16(
        moduli_set[7], moduli_set[6], moduli_set[5], moduli_set[4],
        moduli_set[3], moduli_set[2], moduli_set[1], moduli_set[0]);

    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = 0; j < n; j = j + 2)
        {
            int16_t val = mat[IDX(i, j, n)];
            int16_t val2 = mat[IDX(i, j + 1, n)];
            __m128i a_vec = _mm_set1_epi16(val);
            __m128i a_vec2 = _mm_set1_epi16(val2);

            // Vectorized floored mod
            __m128i res_vec = floored_mod_forward_vec(a_vec, moduli);
            __m128i res_vec2 = floored_mod_forward_vec(a_vec2, moduli);

            // Store to 3D matrix (res_3D_mat[k*(n*n) + i*n + j])
            // Since m == 8, k in [0,7]
            alignas(16) uint16_t tmp[8];

            _mm_store_si128((__m128i *)tmp, res_vec);

            for (int k = 0; k < 8; ++k)
            {
                res_3D_mat[IDX_3D(i, j, k, n)] = static_cast<uint8_t>(tmp[k]);
            }
            _mm_store_si128((__m128i *)tmp, res_vec2);
            for (int k = 0; k < 8; ++k)
            {
                res_3D_mat[IDX_3D(i, j + 1, k, n)] = static_cast<uint8_t>(tmp[k]);
            }
        }
    }
    pthread_exit(NULL);
}

static void forward_RNS_avx_unroll_threadcall(const int16_t *mat, int n,
                                              const uint8_t *moduli_set, int m,
                                              uint8_t *res_3D_mat)
{
    pthread_t threads[NUM_THREADS];
    thread_RNS_forward_data_t thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        thread_data[i].n = n;
        thread_data[i].thread_id = i;
        thread_data[i].mat = &mat;
        thread_data[i].moduli_set = &moduli_set;
        thread_data[i].res_3D_mat = &res_3D_mat;

        pthread_create(&threads[i], NULL, forward_RNS_avx_unroll_worker, (void *)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        pthread_join(threads[i], NULL);
    }
}

static void RNS_MMM_vec(const uint8_t *mat_A_3D, const uint8_t *mat_B_3D,
                        int n, int m, const uint8_t *moduli_set, uint8_t *res_3D_mat)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            __m256i acc = _mm256_setzero_si256(); // 8-lane uint32_t accumulator

            for (int l = 0; l < n; ++l)
            {
                // Load 8 elements (one from each of 8 moduli planes)
                alignas(32) uint32_t a_vec[8], b_vec[8];

                for (int k = 0; k < 8; ++k)
                {
                    a_vec[k] = mat_A_3D[IDX_3D(i, l, k, n)];
                    b_vec[k] = mat_B_3D[IDX_3D(l, j, k, n)];
                }

                __m256i a = _mm256_load_si256((__m256i *)a_vec);
                __m256i b = _mm256_load_si256((__m256i *)b_vec);
                __m256i prod = _mm256_mullo_epi32(a, b);
                acc = _mm256_add_epi32(acc, prod);
            }

            // Now apply modulus to each lane with moduli_set
            alignas(32) uint32_t acc_vals[8];
            _mm256_store_si256((__m256i *)acc_vals, acc);

            for (int k = 0; k < 8; ++k)
            {
                uint32_t sum = acc_vals[k];
                uint8_t mod = moduli_set[k];
                res_3D_mat[IDX_3D(i, j, k, n)] = static_cast<uint8_t>(sum % mod);
            }
        }
    }
}


void *CRT_MMM_worker(void *args)
{
    // Unpack arguments
    thread_CRT_MMM_data_t *data = (thread_CRT_MMM_data_t *)args;
    const uint8_t *res_MM_3D_mat = data->res_MM_3D_mat;
    int64_t *rns_to_int_mat = data->rns_to_int_mat;
    int n = data->n;
    int m = 8;
    int thread_id = data->thread_id;
    int rows_per_thread = n / NUM_THREADS;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id + 1) * rows_per_thread;
    if (thread_id == NUM_THREADS - 1)
    {
        end_row = n; // Last thread takes the remaining rows
    }
    printf("Thread %d processing rows %d to %d\n", thread_id, start_row, end_row);

    const int moduli_set[8] = {254, 253, 251, 249, 247, 245, 241, 239}; //, 233};
    int128_t M_i[8];
    int128_t inv_M_i[8];
    int128_t M = int128_t("13999266705215721930");
    int128_t lambda = (M - 1) / 2;
    M_i[0] = int128_t("55115223248880795");
    M_i[1] = int128_t("55333069981089810");
    M_i[2] = int128_t("55773970937114430");
    M_i[3] = int128_t("56221954639420570");
    M_i[4] = int128_t("56677193138525190");
    M_i[5] = int128_t("57139864102921314");
    M_i[6] = int128_t("58088243590106730");
    M_i[7] = int128_t("58574337678726870");
    inv_M_i[0] = int128_t(191);
    inv_M_i[1] = int128_t(84);
    inv_M_i[2] = int128_t(177);
    inv_M_i[3] = int128_t(196);
    inv_M_i[4] = int128_t(225);
    inv_M_i[5] = int128_t(64);
    inv_M_i[6] = int128_t(227);
    inv_M_i[7] = int128_t(74);
    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int128_t sum = 0;
            for (int k = 0; k < m; ++k)
            {
                uint8_t res_ijk = res_MM_3D_mat[IDX_3D(i, j, k, n)];
                int128_t term = int128_t(res_ijk) * M_i[k] * inv_M_i[k];
                sum += term;
            }
            // Modulo M to wrap around
            int128_t sum_t = floored_mod_CRT(sum, M);
            if (sum_t <= lambda)
            {
                rns_to_int_mat[i * n + j] = static_cast<int64_t>(sum_t);
            }
            else
            {
                rns_to_int_mat[i * n + j] = static_cast<int64_t>(sum_t - M);
            }

            // Save final result as 64-bit integer (should fit safely)
            // rns_to_int_mat[i*n + j] = sum_t;
        }
    }
    pthread_exit(NULL);
}

static void CRT_MMM_threadcall(const uint8_t *res_MM_3D_mat, int n, int m,
                                int64_t *rns_to_int_mat)
{
    pthread_t threads[NUM_THREADS];
    thread_CRT_MMM_data_t thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        thread_data[i].n = n;
        thread_data[i].thread_id = i;
        thread_data[i].res_MM_3D_mat = res_MM_3D_mat;
        thread_data[i].rns_to_int_mat = rns_to_int_mat;

        pthread_create(&threads[i], NULL, CRT_MMM_worker, (void *)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        pthread_join(threads[i], NULL);
    }
}
