/* Neural preconditioner: load binary .precond file and apply SAI, ILU, POLY, or CONVSAI.
 *
 * Binary format (.precond):
 *   Header (40 bytes): magic(u64), n(u64), nnz(u64), mode(u64), reserved_or_nnzU(u64)
 *
 *   mode=0 (ILU): reserved_or_nnzU = nnz_U
 *                  Data = L: row_ptr[n+1], col_idx[nnz_L], values[nnz_L*2]
 *                         U: row_ptr[n+1], col_idx[nnz_U], values[nnz_U*2]
 *   mode=1 (SAI): Data = row_ptr[n+1](u64), col_idx[nnz](u64), values[nnz*2](f64)
 *   mode=2 (POLY): nnz = K+1 (number of coefficients), reserved = 0
 *                   Data = coefficients[(K+1)*2](f64) as interleaved (re,im) pairs
 *   mode=3 (CONVSAI): nnz = n_stencil
 *                   Data = stencil[n_stencil*3](int32), kernel[n_stencil*18](f64)
 *
 * Copyright (C) ADDA contributors
 * This file is part of ADDA.
 */
#include "const.h" // keep this first
#include "precond.h"
#include "comm.h"
#include "fft.h"   // defines FFTW3 macro when FFTW3 is available
#include "io.h"
#include "memory.h"
#include "vars.h"
// system headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef FFTW3
#include <fftw3.h>
#endif

// From fft.c — Dmatrix and its dimensions
extern const doublecomplex * restrict Dmatrix;
extern const size_t DsizeY,DsizeZ;
// From comm.c / vars.h
extern size_t gridX,gridY,gridZ;

bool use_precond=false;
PrecondData precond;

// MatVec from matvec.c — needed for POLY mode Horner evaluation
void MatVec(doublecomplex * restrict in,doublecomplex * restrict out,double * inprod,bool her,
	TIME_TYPE *timing,TIME_TYPE *comm_timing);
// timing variables from timing.c — used for MatVec calls in PolyHorner
extern TIME_TYPE Timing_MVP,Timing_MVPComm;

//======================================================================================================================

/* Helper: read interleaved (re,im) doubles into doublecomplex array */
static void ReadComplexValues(FILE *f,doublecomplex *dest,size_t count,const char *filename,const char *label)
{
	size_t i;
	double *raw=(double *)voidVector(2*count*sizeof(double),ALL_POS,label);
	if (fread(raw,sizeof(double),2*count,f)!=2*count)
		LogError(ONE_POS,"Failed to read %s from preconditioner file '%s'",label,filename);
	for (i=0;i<count;i++)
		dest[i]=raw[2*i]+I*raw[2*i+1];
	free(raw);
}

//======================================================================================================================

static void LoadSAI(FILE *f,const char *filename)
{
	// Allocate SAI arrays
	precond.row_ptr=(uint64_t *)voidVector((precond.n+1)*sizeof(uint64_t),ALL_POS,"precond row_ptr");
	precond.col_idx=(uint64_t *)voidVector(precond.nnz*sizeof(uint64_t),ALL_POS,"precond col_idx");
	precond.values=(doublecomplex *)voidVector(precond.nnz*sizeof(doublecomplex),ALL_POS,"precond values");

	// Read CSR data
	if (fread(precond.row_ptr,sizeof(uint64_t),precond.n+1,f)!=precond.n+1)
		LogError(ONE_POS,"Failed to read row_ptr from preconditioner file '%s'",filename);
	if (fread(precond.col_idx,sizeof(uint64_t),precond.nnz,f)!=precond.nnz)
		LogError(ONE_POS,"Failed to read col_idx from preconditioner file '%s'",filename);
	ReadComplexValues(f,precond.values,precond.nnz,filename,"SAI values");
}

//======================================================================================================================

static void LoadILU(FILE *f,const char *filename)
{
	size_t nnz_L=precond.nnz; // header[2] = nnz_L

	// Allocate L arrays
	precond.L_row_ptr=(uint64_t *)voidVector((precond.n+1)*sizeof(uint64_t),ALL_POS,"precond L_row_ptr");
	precond.L_col_idx=(uint64_t *)voidVector(nnz_L*sizeof(uint64_t),ALL_POS,"precond L_col_idx");
	precond.L_values=(doublecomplex *)voidVector(nnz_L*sizeof(doublecomplex),ALL_POS,"precond L_values");

	// Read L CSR data
	if (fread(precond.L_row_ptr,sizeof(uint64_t),precond.n+1,f)!=precond.n+1)
		LogError(ONE_POS,"Failed to read L row_ptr from preconditioner file '%s'",filename);
	if (fread(precond.L_col_idx,sizeof(uint64_t),nnz_L,f)!=nnz_L)
		LogError(ONE_POS,"Failed to read L col_idx from preconditioner file '%s'",filename);
	ReadComplexValues(f,precond.L_values,nnz_L,filename,"L values");

	// Allocate U arrays
	precond.U_row_ptr=(uint64_t *)voidVector((precond.n+1)*sizeof(uint64_t),ALL_POS,"precond U_row_ptr");
	precond.U_col_idx=(uint64_t *)voidVector(precond.nnz_U*sizeof(uint64_t),ALL_POS,"precond U_col_idx");
	precond.U_values=(doublecomplex *)voidVector(precond.nnz_U*sizeof(doublecomplex),ALL_POS,"precond U_values");

	// Read U CSR data
	if (fread(precond.U_row_ptr,sizeof(uint64_t),precond.n+1,f)!=precond.n+1)
		LogError(ONE_POS,"Failed to read U row_ptr from preconditioner file '%s'",filename);
	if (fread(precond.U_col_idx,sizeof(uint64_t),precond.nnz_U,f)!=precond.nnz_U)
		LogError(ONE_POS,"Failed to read U col_idx from preconditioner file '%s'",filename);
	ReadComplexValues(f,precond.U_values,precond.nnz_U,filename,"U values");
}

//======================================================================================================================

static void LoadFFTDirect(FILE *f,const char *filename)
/* Load FFT-direct preconditioner: Phat stored as 9*gridN complex values.
 * Format: header (mode=4), then gx(u64), gy(u64), gz(u64), then 9*gx*gy*gz interleaved doubles.
 * Reuses CONVSAI apply infrastructure (same Phat layout, same FFT plans).
 */
{
#ifndef FFTW3
	LogError(ONE_POS,"FFTDIRECT preconditioner requires FFTW3");
#else
	uint64_t dims[3];
	size_t gx,gy,gz,gridN;

	/* Read grid dimensions stored after header */
	if (fread(dims,sizeof(uint64_t),3,f)!=3)
		LogError(ONE_POS,"Failed to read grid dims from preconditioner file '%s'",filename);
	gx=(size_t)dims[0];
	gy=(size_t)dims[1];
	gz=(size_t)dims[2];
	gridN=gx*gy*gz;

	/* Verify grid matches ADDA's box (Phat must match the problem) */
	{
		size_t adda_gx=(size_t)fftFit(2*(int)boxX,1);
		size_t adda_gy=(size_t)fftFit(2*(int)boxY,1);
		size_t adda_gz=(size_t)fftFit(2*(int)boxZ,1);
		if (gx!=adda_gx || gy!=adda_gy || gz!=adda_gz)
			LogError(ONE_POS,"FFTDIRECT grid %zux%zux%zu != ADDA grid %zux%zux%zu in '%s'",
				gx,gy,gz,adda_gx,adda_gy,adda_gz,filename);
	}

	precond.conv_gx=gx;
	precond.conv_gy=gy;
	precond.conv_gz=gz;
	precond.conv_gridN=gridN;

	/* Allocate Phat and read directly */
	precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect Phat");
	ReadComplexValues(f,precond.conv_Phat,9*gridN,filename,"FFTDIRECT Phat");

	/* Allocate work buffers */
	precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect work_in");
	precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect work_out");

	/* Create FFTW plans */
	precond.conv_plan_fwd=(void *)fftw_plan_dft_3d(
		(int)gz,(int)gy,(int)gx,
		(fftw_complex *)precond.conv_work_in,
		(fftw_complex *)precond.conv_work_in,
		FFTW_FORWARD,FFTW_MEASURE);
	precond.conv_plan_bwd=(void *)fftw_plan_dft_3d(
		(int)gz,(int)gy,(int)gx,
		(fftw_complex *)precond.conv_work_out,
		(fftw_complex *)precond.conv_work_out,
		FFTW_BACKWARD,FFTW_MEASURE);

	printf("FFTDIRECT preconditioner loaded: grid %zux%zux%zu, %zu Phat values\n",
		gx,gy,gz,9*gridN);
#endif
}

//======================================================================================================================

static void LoadConvSAI(FILE *f,const char *filename)
/* Load ConvSAI preconditioner: stencil displacements + 3×3 complex kernel blocks.
 * Build frequency-domain kernel Phat via FFT for O(N log N) apply.
 */
{
#ifndef FFTW3
	LogError(ONE_POS,"CONVSAI preconditioner requires FFTW3");
#else
	size_t n_stencil=precond.nnz; // nnz field stores n_stencil for CONVSAI
	size_t s,a,b;
	int32_t *stencil_raw;
	doublecomplex *kernel_raw;
	size_t gx,gy,gz,gridN;
	int dx,dy,dz;
	size_t gxi,gyi,gzi,idx;

	// Read stencil displacements (n_stencil × 3 int32)
	stencil_raw=(int32_t *)voidVector(n_stencil*3*sizeof(int32_t),ALL_POS,"convsai stencil");
	if (fread(stencil_raw,sizeof(int32_t),n_stencil*3,f)!=n_stencil*3)
		LogError(ONE_POS,"Failed to read stencil from preconditioner file '%s'",filename);

	// Read kernel values (n_stencil × 9 complex = n_stencil × 18 doubles)
	kernel_raw=(doublecomplex *)voidVector(n_stencil*9*sizeof(doublecomplex),ALL_POS,"convsai kernel");
	ReadComplexValues(f,kernel_raw,n_stencil*9,filename,"CONVSAI kernel");

	// Determine FFT grid: use ADDA's fftFit for FFTW3-optimal sizes (products of 2,3,5,7)
	gx=(size_t)fftFit(2*(int)boxX,1);
	gy=(size_t)fftFit(2*(int)boxY,1);
	gz=(size_t)fftFit(2*(int)boxZ,1);
	gridN=gx*gy*gz;
	precond.conv_gx=gx;
	precond.conv_gy=gy;
	precond.conv_gz=gz;
	precond.conv_gridN=gridN;

	// Allocate frequency-domain kernel: 9 components × gridN
	precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"convsai Phat");

	// Allocate work buffers: 3 components × gridN each
	precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"convsai work_in");
	precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"convsai work_out");

	// Create FFTW plans (in-place, for each component) using FFTW_MEASURE for fast execution
	precond.conv_plan_fwd=(void *)fftw_plan_dft_3d(
		(int)gz,(int)gy,(int)gx,
		(fftw_complex *)precond.conv_work_in,
		(fftw_complex *)precond.conv_work_in,
		FFTW_FORWARD,FFTW_MEASURE);
	precond.conv_plan_bwd=(void *)fftw_plan_dft_3d(
		(int)gz,(int)gy,(int)gx,
		(fftw_complex *)precond.conv_work_out,
		(fftw_complex *)precond.conv_work_out,
		FFTW_BACKWARD,FFTW_MEASURE);

	// Build spatial kernel and FFT to get Phat
	// For each (a,b) pair (9 total): place kernel blocks on grid, then FFT
	{
		doublecomplex *spatial=(doublecomplex *)voidVector(gridN*sizeof(doublecomplex),ALL_POS,"convsai spatial");

		for (a=0;a<3;a++) for (b=0;b<3;b++) {
			// Zero out spatial grid
			memset(spatial,0,gridN*sizeof(doublecomplex));

			// Place kernel entries at stencil positions (with periodic wrapping)
			for (s=0;s<n_stencil;s++) {
				dx=stencil_raw[3*s+0];
				dy=stencil_raw[3*s+1];
				dz=stencil_raw[3*s+2];
				gxi=(size_t)(((int)gx+dx%(int)gx)%(int)gx);
				gyi=(size_t)(((int)gy+dy%(int)gy)%(int)gy);
				gzi=(size_t)(((int)gz+dz%(int)gz)%(int)gz);
				idx=gzi*gy*gx+gyi*gx+gxi;
				spatial[idx]=kernel_raw[9*s+3*a+b];
			}

			// FFT spatial → frequency domain
			fftw_execute_dft((fftw_plan)precond.conv_plan_fwd,
				(fftw_complex *)spatial,(fftw_complex *)spatial);

			// Store in Phat: component (a,b) at offset (3*a+b)*gridN
			memcpy(precond.conv_Phat+(3*a+b)*gridN,spatial,gridN*sizeof(doublecomplex));
		}

		free(spatial);
	}

	free(stencil_raw);
	free(kernel_raw);

	printf("ConvSAI preconditioner loaded: %zu stencil, FFT grid %zux%zux%zu\n",
		n_stencil,gx,gy,gz);
#endif // FFTW3
}

//======================================================================================================================

static void ApplyConvSAI(const doublecomplex *in,doublecomplex *out,size_t n)
/* Apply ConvSAI preconditioner via FFT convolution: out = M * in.
 *
 * Sequential mode:
 *   1. Scatter input dipole vectors onto 3D grid (3 components)
 *   2. FFT forward (3 independent 3D FFTs)
 *   3. Frequency-domain 3×3 matrix multiply with Phat
 *   4. FFT backward (3 independent 3D FFTs)
 *   5. Gather output from grid at dipole positions
 *   6. Normalize by 1/gridN
 *
 * MPI parallel mode:
 *   Same algorithm, but each rank scatters only its local dipoles onto the full 3D grid,
 *   then MPI_Allreduce (sum) combines all contributions. Each rank then independently
 *   performs FFT + multiply + IFFT, and gathers only its local dipole outputs.
 *   This is redundant in compute but simple and correct. The FFT cost O(N log N) is
 *   typically small compared to MatVec communication overhead.
 *
 *   In MPI mode, position[] is local with Z-coordinates relative to local_z0,
 *   so we add local_z0 back when computing grid indices.
 */
{
#ifdef FFTW3
	size_t gx=precond.conv_gx;
	size_t gy=precond.conv_gy;
	size_t gz=precond.conv_gz;
	size_t gridN=precond.conv_gridN;
	doublecomplex *work_in=precond.conv_work_in;
	doublecomplex *work_out=precond.conv_work_out;
	const doublecomplex *Phat=precond.conv_Phat;
	size_t i,comp,ndip;
	int px,py,pz;
	size_t grid_idx;
	double inv_gridN=1.0/(double)gridN;

	ndip=n/3;

	// 1. Zero work buffers
	memset(work_in,0,3*gridN*sizeof(doublecomplex));

	// 2. Scatter: place dipole vectors onto grid
	// In MPI mode, position[].z is relative to local_z0; add offset back for global grid index
	for (i=0;i<ndip;i++) {
		px=position[3*i+0];
		py=position[3*i+1];
		pz=position[3*i+2];
#ifdef PARALLEL
		pz+=local_z0;
#endif
		grid_idx=(size_t)pz*gy*gx+(size_t)py*gx+(size_t)px;
		for (comp=0;comp<3;comp++)
			work_in[comp*gridN+grid_idx]=in[3*i+comp];
	}

#ifdef PARALLEL
	// 2b. MPI: sum scatter contributions from all ranks onto all ranks
	MyInnerProduct(work_in,cmplx_type,3*gridN,NULL);
#endif

	// 3. FFT forward: 3 independent transforms
	for (comp=0;comp<3;comp++)
		fftw_execute_dft((fftw_plan)precond.conv_plan_fwd,
			(fftw_complex *)(work_in+comp*gridN),
			(fftw_complex *)(work_in+comp*gridN));

	// 4. Frequency-domain multiply: out[a] = sum_b Phat[a][b] * in_hat[b]
	// Loop order (a,b,k) for cache-friendly sequential access over k
	memset(work_out,0,3*gridN*sizeof(doublecomplex));
	{
		size_t k;
		size_t a,b;
		for (a=0;a<3;a++) {
			for (b=0;b<3;b++) {
				const doublecomplex *P=Phat+(3*a+b)*gridN;
				const doublecomplex *in_b=work_in+b*gridN;
				doublecomplex *out_a=work_out+a*gridN;
				for (k=0;k<gridN;k++)
					out_a[k]+=P[k]*in_b[k];
			}
		}
	}

	// 5. FFT backward: 3 independent transforms
	for (comp=0;comp<3;comp++)
		fftw_execute_dft((fftw_plan)precond.conv_plan_bwd,
			(fftw_complex *)(work_out+comp*gridN),
			(fftw_complex *)(work_out+comp*gridN));

	// 6. Gather: read results at dipole positions, normalize
	for (i=0;i<ndip;i++) {
		px=position[3*i+0];
		py=position[3*i+1];
		pz=position[3*i+2];
#ifdef PARALLEL
		pz+=local_z0;
#endif
		grid_idx=(size_t)pz*gy*gx+(size_t)py*gx+(size_t)px;
		for (comp=0;comp<3;comp++)
			out[3*i+comp]=work_out[comp*gridN+grid_idx]*inv_gridN;
	}
#endif // FFTW3
}

//======================================================================================================================

static void LoadPoly(FILE *f,const char *filename)
{
	int K=(int)precond.nnz-1; // nnz field stores K+1 for POLY mode
	precond.poly_degree=K;

	// Allocate coefficient array
	precond.poly_coeffs=(doublecomplex *)voidVector((K+1)*sizeof(doublecomplex),ALL_POS,"precond poly_coeffs");
	ReadComplexValues(f,precond.poly_coeffs,K+1,filename,"POLY coefficients");

	// Allocate Horner temp buffer (size n)
	precond.poly_buf=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond poly_buf");
}

//======================================================================================================================

void PrecondLoad(const char *filename)
{
	FILE *f;
	uint64_t header[5];

	f=fopen(filename,"rb");
	if (f==NULL) LogError(ONE_POS,"Failed to open preconditioner file '%s'",filename);

	// Read 40-byte header
	if (fread(header,sizeof(uint64_t),5,f)!=5)
		LogError(ONE_POS,"Failed to read header from preconditioner file '%s'",filename);

	if (header[0]!=PRECOND_MAGIC)
		LogError(ONE_POS,"Invalid magic number in preconditioner file '%s' (expected 0x%lX, got 0x%lX)",
			filename,(unsigned long)PRECOND_MAGIC,(unsigned long)header[0]);

	precond.n=(size_t)header[1];
	precond.nnz=(size_t)header[2];
	precond.mode=header[3];

	if (precond.mode==PRECOND_MODE_SAI) {
		LoadSAI(f,filename);
	} else if (precond.mode==PRECOND_MODE_ILU) {
		precond.nnz_U=(size_t)header[4];
		LoadILU(f,filename);
	} else if (precond.mode==PRECOND_MODE_POLY) {
		LoadPoly(f,filename);
	} else if (precond.mode==PRECOND_MODE_CONVSAI) {
		LoadConvSAI(f,filename);
	} else if (precond.mode==PRECOND_MODE_FFTDIRECT) {
		LoadFFTDirect(f,filename);
	} else {
		LogError(ONE_POS,"Unknown preconditioner mode %lu in file '%s'",(unsigned long)precond.mode,filename);
	}

	// Allocate temporary buffers (needed by all modes for left preconditioning in iterative.c)
	precond.tmp=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp");
	precond.tmp2=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp2");
	precond.r_actual=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond r_actual");
	precond.gather_buf=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond gather_buf");

	fclose(f);
	use_precond=true;
}

//======================================================================================================================

static void SpMV(const uint64_t *row_ptr,const uint64_t *col_idx,const doublecomplex *vals,
                 const doublecomplex *in,doublecomplex *out,size_t n)
/* CSR sparse matrix-vector product: out = A * in (sequential mode, all rows) */
{
	size_t i;
	uint64_t j;

	for (i=0;i<n;i++) {
		doublecomplex sum=0;
		for (j=row_ptr[i];j<row_ptr[i+1];j++)
			sum+=vals[j]*in[col_idx[j]];
		out[i]=sum;
	}
}

//======================================================================================================================

#ifdef PARALLEL
static void SpMV_local(const uint64_t *row_ptr,const uint64_t *col_idx,const doublecomplex *vals,
                       const doublecomplex *in_full,doublecomplex *out_local,size_t row_start,size_t n_local)
/* CSR sparse matrix-vector product for local rows only (MPI mode).
 * in_full is the full gathered vector (size precond.n), out_local is the local output (size n_local).
 * Only rows [row_start, row_start+n_local) are computed.
 */
{
	size_t i;
	uint64_t j;

	for (i=0;i<n_local;i++) {
		doublecomplex sum=0;
		size_t gi=row_start+i;
		for (j=row_ptr[gi];j<row_ptr[gi+1];j++)
			sum+=vals[j]*in_full[col_idx[j]];
		out_local[i]=sum;
	}
}
#endif

//======================================================================================================================

static void SolveLowerTriangular(const uint64_t *row_ptr,const uint64_t *col_idx,const doublecomplex *vals,
                                 const doublecomplex *rhs,doublecomplex *out,size_t n)
/* Forward substitution: solve L*out = rhs where L is lower triangular CSR.
 * L must have non-zero diagonal; diagonal is the last entry in each row (CSR convention).
 */
{
	size_t i;
	uint64_t j;

	for (i=0;i<n;i++) {
		doublecomplex sum=rhs[i];
		doublecomplex diag=0;
		for (j=row_ptr[i];j<row_ptr[i+1];j++) {
			if (col_idx[j]==i)
				diag=vals[j];
			else
				sum-=vals[j]*out[col_idx[j]];
		}
		out[i]=sum/diag;
	}
}

//======================================================================================================================

static void SolveUpperTriangular(const uint64_t *row_ptr,const uint64_t *col_idx,const doublecomplex *vals,
                                 const doublecomplex *rhs,doublecomplex *out,size_t n)
/* Backward substitution: solve U*out = rhs where U is upper triangular CSR.
 * U must have non-zero diagonal; diagonal is the first entry in each row (CSR convention).
 */
{
	size_t i;
	uint64_t j;

	for (i=n;i>0;) {
		i--;
		doublecomplex sum=rhs[i];
		doublecomplex diag=0;
		for (j=row_ptr[i];j<row_ptr[i+1];j++) {
			if (col_idx[j]==i)
				diag=vals[j];
			else
				sum-=vals[j]*out[col_idx[j]];
		}
		out[i]=sum/diag;
	}
}

//======================================================================================================================

static void PolyHorner(const doublecomplex *in,doublecomplex *out,size_t n)
/* Apply polynomial preconditioner p(A)*v via Horner's method.
 *
 * Computes out = p(A)*in = (c_0*I + c_1*A + c_2*A^2 + ... + c_K*A^K) * in
 * using Horner: h = c_K*in; for k=K-1,...,0: h = A*h + c_k*in
 *
 * Uses K calls to MatVec (ADDA's own FFT-based A*v). No extra approximation.
 * poly_buf is used as temporary storage for MatVec output.
 */
{
	int K=precond.poly_degree;
	const doublecomplex *c=precond.poly_coeffs;
	doublecomplex *buf=precond.poly_buf;
	size_t i;
	int k;

	// h = c_K * in
	for (i=0;i<n;i++)
		out[i]=c[K]*in[i];

	// Horner steps: for k = K-1, ..., 0:  h = A*h + c_k*in
	for (k=K-1;k>=0;k--) {
		// buf = A * out  (MatVec: in=out, result=buf)
		MatVec(out,buf,NULL,false,&Timing_MVP,&Timing_MVPComm);
		// out = buf + c_k * in
		for (i=0;i<n;i++)
			out[i]=buf[i]+c[k]*in[i];
	}
}

//======================================================================================================================

void PrecondApply(const doublecomplex *in,doublecomplex *out,size_t n)
/* Apply preconditioner: out = M * in
 * SAI mode:    out = M * in (sparse matrix-vector product)
 * ILU mode:    solve L*z = in, then U*out = z (forward + backward substitution)
 * POLY mode:   out = p(A) * in via Horner's method (K calls to MatVec)
 * CONVSAI/FFT: out = M * in via FFT convolution (scatter + FFT + multiply + IFFT + gather)
 *
 * In MPI parallel mode, 'in' and 'out' are local vectors (size n = local_nRows).
 * SAI mode:    AllGather full input, SpMV on local rows only.
 * CONVSAI/FFT: Each rank scatters local dipoles to full grid, AllReduce to sum contributions,
 *              then each rank independently does FFT+multiply+IFFT and gathers local results.
 */
{
	if (precond.mode==PRECOND_MODE_SAI) {
#ifdef PARALLEL
		/* MPI: in has only local_nRows elements but SpMV needs the full vector (col_idx uses global indices).
		 * Gather full input vector, then compute only local rows of the SpMV product.
		 */
		AllGather((void *)in,precond.gather_buf,cmplx3_type,NULL);
		SpMV_local(precond.row_ptr,precond.col_idx,precond.values,
		           precond.gather_buf,out,3*local_nvoid_d0,n);
#else
		SpMV(precond.row_ptr,precond.col_idx,precond.values,in,out,n);
#endif
	} else if (precond.mode==PRECOND_MODE_POLY) {
		PolyHorner(in,out,n);
	} else if (precond.mode==PRECOND_MODE_CONVSAI || precond.mode==PRECOND_MODE_FFTDIRECT) {
		ApplyConvSAI(in,out,n);
	} else {
		// ILU: solve L*z = in, then U*out = z
		doublecomplex *z=(doublecomplex *)voidVector(n*sizeof(doublecomplex),ALL_POS,"precond ILU z");
		SolveLowerTriangular(precond.L_row_ptr,precond.L_col_idx,precond.L_values,in,z,n);
		SolveUpperTriangular(precond.U_row_ptr,precond.U_col_idx,precond.U_values,z,out,n);
		free(z);
	}
}

//======================================================================================================================

void PrecondApplyScaled(const doublecomplex *in,doublecomplex *out,size_t n)
/* Apply preconditioner with ADDA's diagonal scaling correction:
 *   out = S * M * S^{-1} * in
 * where S = diag(cc_sqrt[material[i]][component]).
 *
 * The preconditioner was trained on the raw DDA matrix A = I - alpha*G, but ADDA's iterative solver
 * operates on the scaled system (I + S*D*S). To bridge the mismatch, we transform from ADDA's scaled
 * space back to the raw space (multiply by S^{-1}), apply M, then transform back (multiply by S).
 */
{
	size_t i,dip;
	int comp;
	doublecomplex s;
	doublecomplex *buf;

	buf=(doublecomplex *)voidVector(n*sizeof(doublecomplex),ALL_POS,"precond scale buf");

	// Step 1: buf = S^{-1} * in  (undo ADDA's sqrt(C) scaling)
	for (i=0;i<n;i++) {
		dip=i/3;
		comp=(int)(i%3);
		s=cc_sqrt[material[dip]][comp];
		buf[i]=in[i]/s;
	}
	// Step 2: out = M * buf  (apply preconditioner in raw space)
	PrecondApply(buf,out,n);
	// Step 3: out = S * out  (transform back to ADDA's scaled space)
	for (i=0;i<n;i++) {
		dip=i/3;
		comp=(int)(i%3);
		s=cc_sqrt[material[dip]][comp];
		out[i]=s*out[i];
	}
	free(buf);
}

//======================================================================================================================

void PrecondFree(void)
{
	if (use_precond) {
		if (precond.mode==PRECOND_MODE_SAI) {
			free(precond.row_ptr);
			free(precond.col_idx);
			free(precond.values);
		} else if (precond.mode==PRECOND_MODE_POLY) {
			free(precond.poly_coeffs);
			free(precond.poly_buf);
		} else if (precond.mode==PRECOND_MODE_CONVSAI || precond.mode==PRECOND_MODE_FFTDIRECT) {
#ifdef FFTW3
			fftw_destroy_plan((fftw_plan)precond.conv_plan_fwd);
			fftw_destroy_plan((fftw_plan)precond.conv_plan_bwd);
#endif
			free(precond.conv_Phat);
			free(precond.conv_work_in);
			free(precond.conv_work_out);
		} else {
			free(precond.L_row_ptr);
			free(precond.L_col_idx);
			free(precond.L_values);
			free(precond.U_row_ptr);
			free(precond.U_col_idx);
			free(precond.U_values);
		}
		free(precond.tmp);
		free(precond.tmp2);
		free(precond.r_actual);
		free(precond.gather_buf);
		use_precond=false;
	}
}

//======================================================================================================================

void DumpDhat(const char *filename)
/* Dump ADDA's frequency-domain D_hat (Dmatrix) as a binary file.
 *
 * Dmatrix is stored in reduced form: 6 components (upper triangle of symmetric 3x3),
 * only y < DsizeY = gridY/2+1, z < DsizeZ = gridZ/2+1.
 * Index: NDCOMP*((x*DsizeZ+z)*DsizeY+y) + comp
 *
 * Output format: full 6 × gridX × gridY × gridZ complex, with symmetry expanded.
 * D_hat has symmetry: D(x, gridY-y, gridZ-z) = D(x, y, z) for the diagonal components,
 * and sign flips for off-diagonal (xy: flip y, xz: flip z, yz: flip y&z).
 *
 * File layout:
 *   gridX(u64), gridY(u64), gridZ(u64)
 *   6 * gridX * gridY * gridZ interleaved (re,im) doubles
 *   Component order: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
 *
 * The sign convention for off-diagonal elements when reflected:
 *   xy (comp 1): negate when y is reflected
 *   xz (comp 2): negate when z is reflected
 *   yz (comp 4): negate when y XOR z is reflected
 */
{
	FILE *f;
	uint64_t dims[3];
	size_t x,y,z,comp;
	size_t gridN;
	doublecomplex *full;
	size_t yr,zr;
	int y_refl,z_refl;
	doublecomplex val;
	double sign;

	f=fopen(filename,"wb");
	if (f==NULL) LogError(ONE_POS,"Failed to create D_hat dump file '%s'",filename);

	dims[0]=(uint64_t)gridX;
	dims[1]=(uint64_t)gridY;
	dims[2]=(uint64_t)gridZ;
	fwrite(dims,sizeof(uint64_t),3,f);

	gridN=gridX*gridY*gridZ;
	full=(doublecomplex *)malloc(gridN*sizeof(doublecomplex));

	for (comp=0;comp<6;comp++) {
		/* Expand reduced Dmatrix to full grid for this component */
		for (x=0;x<gridX;x++) for (y=0;y<gridY;y++) for (z=0;z<gridZ;z++) {
			/* Map (y,z) to reduced range */
			yr=y; zr=z;
			y_refl=0; z_refl=0;
			if (yr>=DsizeY) { yr=gridY-yr; y_refl=1; }
			if (zr>=DsizeZ) { zr=gridZ-zr; z_refl=1; }

			val=Dmatrix[6*((x*DsizeZ+zr)*DsizeY+yr)+comp];

			/* Apply sign flip for off-diagonal components */
			sign=1.0;
			if (comp==1 && y_refl) sign=-1.0; /* xy: flip on y reflection */
			if (comp==2 && z_refl) sign=-1.0; /* xz: flip on z reflection */
			if (comp==4) { /* yz: flip on y XOR z reflection */
				if (y_refl!=z_refl) sign=-1.0;
			}
			full[z*gridY*gridX+y*gridX+x]=val*sign;
		}

		/* Write as interleaved re,im */
		{
			double *raw=(double *)malloc(2*gridN*sizeof(double));
			size_t i;
			for (i=0;i<gridN;i++) {
				raw[2*i]=creal(full[i]);
				raw[2*i+1]=cimag(full[i]);
			}
			fwrite(raw,sizeof(double),2*gridN,f);
			free(raw);
		}
	}

	free(full);
	fclose(f);
	printf("D_hat dumped: %zux%zux%zu, 6 components -> %s\n",gridX,gridY,gridZ,filename);
}
