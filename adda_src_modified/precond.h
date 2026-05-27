/* Neural preconditioner support for ADDA (SAI, ILU, POLY, and CONVSAI modes)
 *
 * Loads a preconditioner from a binary .precond file and applies it as left preconditioning
 * in BiCGStab: solve M*A*x = M*b.
 *
 * Four modes:
 *   mode=0 (ILU): L (lower) and U (upper) triangular factors. Apply via forward+backward solve:
 *                  solve L*z = in, then U*out = z. This computes out = (L*U)^{-1} * in.
 *   mode=1 (SAI): M is a general sparse matrix. Apply via SpMV: out = M * in.
 *   mode=2 (POLY): Polynomial preconditioner p(A)*v. K+1 complex coefficients, applied via
 *                   Horner's method using ADDA's own MatVec. Zero extra memory beyond coefficients.
 *   mode=3 (CONVSAI): Translation-invariant convolution kernel. Applied via FFT convolution for large kernels
 *                      or direct sparse convolution for small kernels.
 *
 * The actual residual ||b - A*x|| / ||b|| is tracked algebraically using intermediate A*p and A*s
 * values (before M is applied), so no extra MatVec is needed per iteration.
 */
#ifndef __precond_h
#define __precond_h

#include "types.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define PRECOND_MAGIC 0x4E49464C
#define PRECOND_MODE_ILU 0
#define PRECOND_MODE_SAI 1
#define PRECOND_MODE_POLY 2
#define PRECOND_MODE_CONVSAI 3
#define PRECOND_MODE_FFTDIRECT 4

typedef struct {
	size_t n;
	size_t nnz;
	uint64_t mode;
	/* SAI mode: single matrix M */
	uint64_t *row_ptr;
	uint64_t *col_idx;
	doublecomplex *values;
	/* ILU mode: L (lower) and U (upper) triangular factors */
	size_t nnz_U;
	uint64_t *L_row_ptr;
	uint64_t *L_col_idx;
	doublecomplex *L_values;
	uint64_t *U_row_ptr;
	uint64_t *U_col_idx;
	doublecomplex *U_values;
	/* POLY mode: polynomial preconditioner p(A)*v via Horner */
	int poly_degree;            // K: polynomial degree
	doublecomplex *poly_coeffs; // K+1 complex coefficients c_0...c_K
	doublecomplex *poly_buf;    // temp buffer (size n) for Horner evaluation
	/* CONVSAI mode: convolution preconditioner */
	size_t conv_gx,conv_gy,conv_gz; // FFT grid dimensions (= 2*boxX, 2*boxY, 2*boxZ padded)
	size_t conv_gridN;               // = conv_gx * conv_gy * conv_gz
	bool conv_direct;                // use direct sparse convolution instead of FFT
	size_t conv_n_stencil;           // number of stencil offsets for CONVSAI
	int32_t *conv_stencil;           // direct mode: n_stencil * 3 displacement vectors
	doublecomplex *conv_kernel;      // direct mode: n_stencil * 9 complex kernel blocks
	int64_t *conv_grid_to_dipole;    // direct mode: grid index -> global dipole index, -1 for empty
	size_t conv_direct_edges;        // direct mode: number of precomputed local neighbor entries
	size_t *conv_direct_row_ptr;     // direct mode: local dipole -> neighbor entry range
	uint64_t *conv_direct_dipole;    // direct mode: neighbor entry -> global dipole index
	uint32_t *conv_direct_stencil;   // direct mode: neighbor entry -> stencil index
	doublecomplex *conv_Phat;        // frequency-domain kernel: 9 * conv_gridN complex values
	doublecomplex *conv_work_in;     // work buffer for input:  3 * conv_gridN
	doublecomplex *conv_work_out;    // work buffer for output: 3 * conv_gridN
	void *conv_plan_fwd;             // fftw_plan forward (cast to void* to avoid fftw3.h in header)
	void *conv_plan_bwd;             // fftw_plan backward
	/* Temporary buffers */
	doublecomplex *tmp;     // temp buffer: stores A*p intermediate in left preconditioning
	doublecomplex *tmp2;    // temp buffer: stores A*s intermediate in left preconditioning
	doublecomplex *r_actual; // tracks actual residual b - A*x for convergence monitoring
	doublecomplex *gather_buf; // MPI: full-size buffer for AllGather before SpMV
} PrecondData;

extern bool use_precond;
extern PrecondData precond;

void PrecondLoad(const char *filename);
void PrecondApply(const doublecomplex *in,doublecomplex *out,size_t n);
void PrecondApplyScaled(const doublecomplex *in,doublecomplex *out,size_t n);
void PrecondFree(void);
void DumpDhat(const char *filename);

#endif // __precond_h
