/* quant_matmul.c – simple C version with typedef data_t
 *   (c) 2025 Sina Karimi – public‑domain / CC0
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cmath> // For std::round
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <cmath> // For std::round
#include <boost/multiprecision/cpp_int.hpp> // For int128_t
using namespace boost::multiprecision;

/* =====================================================
 * 1.  Choose ONE line – e.g. float or double or _Float16
 * =====================================================*/
typedef float data_t;        /* change here if you need BF16 / FP64 / etc. */

/* Handy macro for row‑major indexing of a flat array */
#define IDX(i,j,n)  ((i)*(n) + (j))
#define IDX_3D(i,j,k,n)  ((i)*(n) + (j) + k*(n*n))
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
static void matmul_float(const float *A, const float *B,
                         int n, float *C);
static void dequantise(const int64_t *Cq, int n,
                       const data_t *scaleA, const data_t *scaleB,
                       data_t *C);

int128_t gcdExtended(int128_t a, int128_t b, int128_t* x,int128_t* y);
int128_t modInverse(int128_t A, int128_t M) ;
uint8_t floored_mod_forward(int16_t a, uint8_t b);
static void forward_RNS(const int16_t *mat,int n,       // int n is the size of the matrix
                        const uint8_t *moduli_set,int m, // int m is the size of the moduli set 
                        uint8_t *res_3D_mat);
uint8_t floored_mod_MM(uint32_t a, uint8_t b);
static void RNS_MMM(const uint8_t *mat_A_3D, const uint8_t *mat_B_3D, 
                     int n,  int m, const uint8_t *moduli_set,
                    uint8_t *res_3D_mat) ;
int128_t floored_mod_CRT(int128_t a, int128_t b);
 static void CRT_MMM(const uint8_t *res_MM_3D_mat, int n, int m,
                    int64_t *rns_to_int_mat );






/* ===========================================================================*/

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int n = atoi(argv[1]);
    if (n <= 0) { fputs("matrix_size must be positive.\n", stderr); return 1; }

    srand((unsigned)time(NULL));

    /* Moduli set */
    const uint8_t moduli_set[9] = {254, 253, 251, 249, 247, 245, 241, 239, 233};
    const int m = 9;  // 9 moduli

    /* 1. allocate & fill FP32 input matrices ------------------------------ */
    data_t *A = alloc_matrix(n);
    data_t *B = alloc_matrix(n);
    data_t *C_res = alloc_matrix(n);
    init_random(A, n);
    init_random(B, n);
    
    matmul_float(A,B,n,C_res);
      /* 8. print result ------------------------------------------------------ */
    printf("\n input result:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%8.4f ", C_res[IDX(i, j, n)]);
        putchar('\n');
    }

    /* 2. scaling factors -------------------------------------------------- */
    data_t *scaleA = (data_t *)malloc(n * sizeof(data_t));
    data_t *scaleB = (data_t *)malloc(n * sizeof(data_t));
    scale_rows(A, n, scaleA);
    scale_cols(B, n, scaleB);

    /* 3. quantise to INT16 ------------------------------------------------ */
    int16_t *Aq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    int16_t *Bq = (int16_t *)malloc((size_t)n * n * sizeof(int16_t));
    fp32_to_int16(A, n, scaleA, NULL, 0, Aq);   /* row-wise */
    fp32_to_int16(B, n, NULL, scaleB, 1, Bq);   /* col-wise */

    /* 4. forward RNS conversion ------------------------------------------- */
    uint8_t *Aq_3D = (uint8_t *)malloc((size_t)n * n * m * sizeof(uint8_t));
    uint8_t *Bq_3D = (uint8_t *)malloc((size_t)n * n * m * sizeof(uint8_t));
    forward_RNS(Aq, n, moduli_set, m, Aq_3D);
    forward_RNS(Bq, n, moduli_set, m, Bq_3D);

    /* 5. RNS MMM ----------------------------------------------------------- */
    uint8_t *Cq_3D = (uint8_t *)malloc((size_t)n * n * m * sizeof(uint8_t));
    RNS_MMM(Aq_3D, Bq_3D, n, m,moduli_set, Cq_3D);

    /* 6. CRT reconstruction ------------------------------------------------ */
    int64_t *Cq = (int64_t *)malloc((size_t)n * n * sizeof(int64_t));
    CRT_MMM(Cq_3D, n, m, Cq);

    /* 7. Dequantise -------------------------------------------------------- */
    data_t *C = alloc_matrix(n);
    dequantise(Cq, n, scaleA, scaleB, C);

    /* 8. print result ------------------------------------------------------ */
    printf("\nResult matrix (after RNS MatMul + CRT + Dequantization):\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%8.4f ", C[IDX(i, j, n)]);
        putchar('\n');
    }

    /* 9. cleanup ----------------------------------------------------------- */
    free(A);  free(B);  free(C);
    free(Aq); free(Bq); free(Cq);
    free(Aq_3D); free(Bq_3D); free(Cq_3D);
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


static void matmul_float(const float *A, const float *B,
                         int n, float *C)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
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
    const data_t   inv = 1.0f / ((data_t)qmax * (data_t)qmax);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[IDX(i, j, n)] = (data_t)Cq[IDX(i, j, n)] *
                              scaleA[i] * scaleB[j] * inv;
}




/*--------Utility Functions for pre-computed constants----------*/
int128_t gcdExtended(int128_t a, int128_t b, int128_t* x,int128_t* y)
{

    // Base Case
    if (a == 0) {
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
int128_t modInverse(int128_t A, int128_t M) {
  
      
    for (int128_t X = 1; X < M; X++)
        if ((A*X)%M == 1)
            return X;
}





/*---------------------------------------------------------------*/
uint8_t floored_mod_forward(int16_t a, uint8_t b) { // to be used in forward RNS 
    return static_cast<uint8_t>( ((a % static_cast<int16_t>(b)) + static_cast<int16_t>(b)) % static_cast<int16_t>(b));
}

static void forward_RNS(const int16_t *mat,int n,       // int n is the size of the matrix
                        const uint8_t *moduli_set,int m, // int m is the size of the moduli set 
                        uint8_t *res_3D_mat)
{
    /*
    Calculates residues for each element of the matrix.
    Returns a 3D matrix each element is a 1x8 vector containing RNS representation of the corresponding element. 
    */
    //int8_t *res_vector = (int8_t *)malloc(n * sizeof(int8_t));
    // First calculate residues using floored modulus
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; ++j) {
            for(int k=0;k<m;k++){
                res_3D_mat[IDX_3D(i,j,k,n)] = floored_mod_forward((mat[IDX(i, j, n)]), moduli_set[k]);
            }
           
        }
    }
    // res_3D_mat =[8xNxN] --> res_3D_mat[0*(n*n) + 0*n + 0] ,
}
uint8_t floored_mod_MM(uint32_t a, uint8_t b) { // to be used in forward RNS 
    return static_cast<uint8_t>( ((a % static_cast<uint32_t>(b)) + static_cast<uint32_t>(b)) % static_cast<uint32_t>(b));
}
static void RNS_MMM(const uint8_t *mat_A_3D, const uint8_t *mat_B_3D, int n, int m, const uint8_t *moduli_set,uint8_t *res_3D_mat)       //nxnxm matrices where m=size of the moduli set
                                              //                     n=size of the square matrices
{
    /*
    Performs matrix matrix multiplication in RNS domain, 3D matrix multiplication WITH mod operation.
    Each "layer" corresponding to a modulus is independent. 
    For TC, we need to discard mod thingie mingie here.

    */
   for (int k = 0; k < m; ++k) { // For each modulus
        for (int i = 0; i < n; ++i) { // For each row of A
            for (int j = 0; j < n; ++j) { // For each column of B
                uint32_t sum = 0;
                for (int l = 0; l < n; ++l) { // Sum over row of A and column of B
                    uint32_t a = mat_A_3D[IDX_3D(i, l, k, n)];
                    uint32_t b = mat_B_3D[IDX_3D(l, j, k, n)];
                    sum += a * b; //mod thingie mingie here
                }
                res_3D_mat[IDX_3D(i, j, k, n)] = floored_mod_MM(sum, moduli_set[k]); // You can remove mod if needed as res_3D_mat[IDX_3D(i, j, k, n)] = sum;

            }
        }
    }
}

int128_t floored_mod_CRT(int128_t a, int128_t b) {
    return static_cast<int128_t>( ((a % b) + b) % b);
}

static void CRT_MMM(const uint8_t *res_MM_3D_mat, int n,  int m,
                    int64_t *rns_to_int_mat )
{
    /*
    m={254,253,251,249,247,245,241,239,233} i=0,...,8
    We need to pre-calculate intermediate values such that  M=prod(m_i), M_i=M\m_i without overflow, i.e. we need 72-bit for M AND 64-bit for M_i's
    I am using int128_t for all for now

    res_MM_3D_mat: (n x n x m) flattened 3D matrix, where each element is 8-bit residue
    rns_to_int_mat: output (n x n) flattened 2D matrix, where each element is 64-bit normal integer

    */

   
   // Moduli array (your m_i's)
    const int moduli_set[9] = {254, 253, 251, 249, 247, 245, 241, 239, 233};

    // Precomputed M and M_i values--- declare as constants once calcualting them
    int128_t M = int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249) * int128_t(247) * int128_t(245) * int128_t(241) * int128_t(239) * int128_t(233);
    int128_t lambda= (M-1)/2;
    int128_t M_i[9];
    for (int k = 0; k < 9; ++k) {
        M_i[k] = 1;
        for (int l = 0; l < 9; ++l) {
            if (l != k) {
                M_i[k] *= int128_t(moduli_set[l]);
            }
        }
    }
    int128_t inv_M_i[9];
    for (int k = 0; k < 9; ++k) {
        inv_M_i[k] = modInverse(M_i[k], int128_t(moduli_set[k]));
    }
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
    int128_t M_8= int128_t(254) * int128_t(253) * int128_t(251) * int128_t(249)* int128_t(247)* int128_t(245)* int128_t(241)* int128_t(239);

    int128_t M_i[9]={M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8}
    */
    // Precompute modular inverses mod m_i
    // Now for each (i,j) position do CRT reconstruction
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int128_t sum = 0;
            for (int k = 0; k < m; ++k) {
                uint8_t res_ijk = res_MM_3D_mat[IDX_3D(i,j,k,n)];
                int128_t term = int128_t(res_ijk) * M_i[k] * inv_M_i[k];
                sum += term;
            }
            // Modulo M to wrap around
            int128_t sum_t = floored_mod_CRT(sum, M);
            if(sum_t<=lambda){
                rns_to_int_mat[i*n + j] = static_cast<int64_t>(sum_t);
            }else{
                rns_to_int_mat[i*n + j] = static_cast<int64_t>(sum_t - M);
            }

            // Save final result as 64-bit integer (should fit safely)
            //rns_to_int_mat[i*n + j] = sum_t;
        }
    }
    
}