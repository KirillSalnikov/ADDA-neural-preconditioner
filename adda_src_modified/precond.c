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
#ifndef SPARSE
extern doublecomplex * restrict Xmatrix,* restrict slices,* restrict slices_tr;
#endif
// From comm.c / vars.h
extern size_t gridX,gridY,gridZ;

bool use_precond=false;
PrecondData precond;

#define CONVSAI_DIRECT_DEFAULT_MAX_STENCIL 8192
#define CONVSAI_DIRECT_PRECOMPUTE_DEFAULT_MB 256

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

static bool ParseBoolEnv(const char *value,bool *out)
{
	if (value==NULL || value[0]=='\0') return false;
	if (strcmp(value,"1")==0 || strcmp(value,"true")==0 || strcmp(value,"TRUE")==0 ||
	    strcmp(value,"yes")==0 || strcmp(value,"YES")==0 || strcmp(value,"on")==0 ||
	    strcmp(value,"ON")==0) {
		*out=true;
		return true;
	}
	if (strcmp(value,"0")==0 || strcmp(value,"false")==0 || strcmp(value,"FALSE")==0 ||
	    strcmp(value,"no")==0 || strcmp(value,"NO")==0 || strcmp(value,"off")==0 ||
	    strcmp(value,"OFF")==0) {
		*out=false;
		return true;
	}
	return false;
}

static bool ShouldUseConvSAIDirect(size_t n_stencil)
{
	bool forced;
	const char *env=getenv("ADDA_CONVSAI_DIRECT");
	if (ParseBoolEnv(env,&forced)) return forced;

	env=getenv("ADDA_CONVSAI_DIRECT_MAX_STENCIL");
	if (env!=NULL && env[0]!='\0') {
		char *endptr=NULL;
		unsigned long max_stencil=strtoul(env,&endptr,10);
		if (endptr!=env && *endptr=='\0')
			return n_stencil<=(size_t)max_stencil;
	}
	return n_stencil<=CONVSAI_DIRECT_DEFAULT_MAX_STENCIL;
}

static bool ShouldPrecomputeConvSAIDirect(size_t n_stencil)
{
	bool forced;
	const char *env=getenv("ADDA_CONVSAI_DIRECT_PRECOMPUTE");
	if (ParseBoolEnv(env,&forced)) return forced;

	size_t budget_mb=CONVSAI_DIRECT_PRECOMPUTE_DEFAULT_MB;
	env=getenv("ADDA_CONVSAI_DIRECT_PRECOMPUTE_MB");
	if (env!=NULL && env[0]!='\0') {
		char *endptr=NULL;
		unsigned long parsed=strtoul(env,&endptr,10);
		if (endptr!=env && *endptr=='\0') budget_mb=(size_t)parsed;
	}

	long double bytes=(long double)local_nvoid_Ndip*(long double)n_stencil*
		(long double)(sizeof(uint64_t)+sizeof(uint32_t));
	bytes+=(long double)(local_nvoid_Ndip+1)*sizeof(size_t);
	return bytes<=(long double)budget_mb*1024.0L*1024.0L;
}

static bool ShouldUseConvSAIDistributed(void)
{
	bool forced;
	const char *env=getenv("ADDA_CONVSAI_DISTRIBUTED");
	if (ParseBoolEnv(env,&forced)) return forced;
#ifdef PARALLEL
	return true;
#else
	return false;
#endif
}

static size_t WrapConvCoord(long long value,size_t dim)
{
	long long d=(long long)dim;
	long long r=value%d;
	if (r<0) r+=d;
	return (size_t)r;
}

static size_t ConvGridIndex(size_t x,size_t y,size_t z)
{
	return z*precond.conv_gy*precond.conv_gx+y*precond.conv_gx+x;
}

static int64_t ConvSAIDirectNeighborAt(int px,int py,int pz,size_t stencil_id)
{
	size_t qx=WrapConvCoord((long long)px-(long long)precond.conv_stencil[3*stencil_id+0],precond.conv_gx);
	size_t qy=WrapConvCoord((long long)py-(long long)precond.conv_stencil[3*stencil_id+1],precond.conv_gy);
	size_t qz=WrapConvCoord((long long)pz-(long long)precond.conv_stencil[3*stencil_id+2],precond.conv_gz);
	return precond.conv_grid_to_dipole[ConvGridIndex(qx,qy,qz)];
}

static void BuildConvSAIDirectMap(void)
{
	size_t i,grid_idx;
	int *pos_global;
	precond.conv_grid_to_dipole=(int64_t *)voidVector(
		precond.conv_gridN*sizeof(int64_t),ALL_POS,"convsai grid-to-dipole map");
	for (i=0;i<precond.conv_gridN;i++) precond.conv_grid_to_dipole[i]=-1;

#ifdef PARALLEL
	{
		size_t local_len=3*local_nvoid_Ndip;
		int *pos_local=(int *)voidVector((local_len ? local_len : 1)*sizeof(int),ALL_POS,
			"convsai local positions");
		size_t j;
		for (j=0;j<local_nvoid_Ndip;j++) {
			pos_local[3*j+0]=(int)position[3*j+0];
			pos_local[3*j+1]=(int)position[3*j+1];
			pos_local[3*j+2]=(int)position[3*j+2]+local_z0;
		}
		pos_global=(int *)voidVector(3*nvoid_Ndip*sizeof(int),ALL_POS,"convsai global positions");
		AllGather(pos_local,pos_global,int3_type,NULL);
		free(pos_local);
	}
#else
	pos_global=(int *)voidVector(3*nvoid_Ndip*sizeof(int),ALL_POS,"convsai global positions");
	for (i=0;i<nvoid_Ndip;i++) {
		pos_global[3*i+0]=(int)position[3*i+0];
		pos_global[3*i+1]=(int)position[3*i+1];
		pos_global[3*i+2]=(int)position[3*i+2];
	}
#endif

	for (i=0;i<nvoid_Ndip;i++) {
		int px=pos_global[3*i+0];
		int py=pos_global[3*i+1];
		int pz=pos_global[3*i+2];
		if (px<0 || py<0 || pz<0 ||
		    (size_t)px>=precond.conv_gx || (size_t)py>=precond.conv_gy || (size_t)pz>=precond.conv_gz)
			LogError(ALL_POS,"Particle position (%d,%d,%d) is outside ConvSAI grid %zux%zux%zu",
				px,py,pz,precond.conv_gx,precond.conv_gy,precond.conv_gz);
		grid_idx=ConvGridIndex((size_t)px,(size_t)py,(size_t)pz);
		precond.conv_grid_to_dipole[grid_idx]=(int64_t)i;
	}
	free(pos_global);
}

static void BuildConvSAIDirectNeighbors(void)
{
	size_t i,s,total,entry;

	precond.conv_direct_edges=0;
	precond.conv_direct_row_ptr=NULL;
	precond.conv_direct_dipole=NULL;
	precond.conv_direct_stencil=NULL;

	if (!ShouldPrecomputeConvSAIDirect(precond.conv_n_stencil)) return;

	precond.conv_direct_row_ptr=(size_t *)voidVector(
		(local_nvoid_Ndip+1)*sizeof(size_t),ALL_POS,"convsai direct row_ptr");

	total=0;
	precond.conv_direct_row_ptr[0]=0;
	for (i=0;i<local_nvoid_Ndip;i++) {
		int px=position[3*i+0];
		int py=position[3*i+1];
		int pz=position[3*i+2];
#ifdef PARALLEL
		pz+=local_z0;
#endif
		for (s=0;s<precond.conv_n_stencil;s++)
			if (ConvSAIDirectNeighborAt(px,py,pz,s)>=0) total++;
		precond.conv_direct_row_ptr[i+1]=total;
	}

	precond.conv_direct_edges=total;
	precond.conv_direct_dipole=(uint64_t *)voidVector(
		(total ? total : 1)*sizeof(uint64_t),ALL_POS,"convsai direct dipoles");
	precond.conv_direct_stencil=(uint32_t *)voidVector(
		(total ? total : 1)*sizeof(uint32_t),ALL_POS,"convsai direct stencils");

	entry=0;
	for (i=0;i<local_nvoid_Ndip;i++) {
		int px=position[3*i+0];
		int py=position[3*i+1];
		int pz=position[3*i+2];
#ifdef PARALLEL
		pz+=local_z0;
#endif
		for (s=0;s<precond.conv_n_stencil;s++) {
			int64_t dip=ConvSAIDirectNeighborAt(px,py,pz,s);
			if (dip>=0) {
				precond.conv_direct_dipole[entry]=(uint64_t)dip;
				precond.conv_direct_stencil[entry]=(uint32_t)s;
				entry++;
			}
		}
	}

	if (IFROOT)
		printf("ConvSAI direct neighbor cache: %zu local entries (%.1f avg per local dipole)\n",
			total,local_nvoid_Ndip ? (double)total/(double)local_nvoid_Ndip : 0.0);
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
 * Reuses CONVSAI apply infrastructure (same Phat layout).
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

	/* Verify grid matches ADDA's actual FFT grid, including MPI divisibility constraints. */
	{
		size_t adda_gx=gridX;
		size_t adda_gy=gridY;
		size_t adda_gz=gridZ;
		if (gx!=adda_gx || gy!=adda_gy || gz!=adda_gz)
			LogError(ONE_POS,"FFTDIRECT grid %zux%zux%zu != ADDA grid %zux%zux%zu in '%s'",
				gx,gy,gz,adda_gx,adda_gy,adda_gz,filename);
	}

	precond.conv_gx=gx;
	precond.conv_gy=gy;
	precond.conv_gz=gz;
	precond.conv_gridN=gridN;
	precond.conv_direct=false;
	precond.conv_n_stencil=0;
	precond.conv_stencil=NULL;
	precond.conv_kernel=NULL;
	precond.conv_grid_to_dipole=NULL;
	precond.conv_direct_edges=0;
	precond.conv_direct_row_ptr=NULL;
	precond.conv_direct_dipole=NULL;
	precond.conv_direct_stencil=NULL;
	precond.conv_Phat=NULL;
	precond.conv_work_in=NULL;
	precond.conv_work_out=NULL;
	precond.conv_plan_fwd=NULL;
	precond.conv_plan_bwd=NULL;

	/* Allocate Phat and read directly */
	precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect Phat");
	ReadComplexValues(f,precond.conv_Phat,9*gridN,filename,"FFTDIRECT Phat");

	if (!ShouldUseConvSAIDistributed()) {
		/* Allocate work buffers for the replicated FFT fallback. */
		precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect work_in");
		precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect work_out");

		/* Create FFTW plans for the replicated FFT fallback. */
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
	}

	if (IFROOT)
		printf("FFTDIRECT preconditioner loaded: grid %zux%zux%zu, %zu Phat values%s\n",
			gx,gy,gz,9*gridN,ShouldUseConvSAIDistributed() ? ", distributed MPI apply" : "");
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

	// Use ADDA's actual FFT grid, including MPI divisibility constraints.
	gx=gridX;
	gy=gridY;
	gz=gridZ;
	gridN=gx*gy*gz;
	precond.conv_gx=gx;
	precond.conv_gy=gy;
	precond.conv_gz=gz;
	precond.conv_gridN=gridN;
	precond.conv_n_stencil=n_stencil;
	precond.conv_stencil=NULL;
	precond.conv_kernel=NULL;
	precond.conv_grid_to_dipole=NULL;
	precond.conv_direct_edges=0;
	precond.conv_direct_row_ptr=NULL;
	precond.conv_direct_dipole=NULL;
	precond.conv_direct_stencil=NULL;
	precond.conv_direct=ShouldUseConvSAIDirect(n_stencil);
	precond.conv_Phat=NULL;
	precond.conv_work_in=NULL;
	precond.conv_work_out=NULL;
	precond.conv_plan_fwd=NULL;
	precond.conv_plan_bwd=NULL;

	if (precond.conv_direct) {
		precond.conv_stencil=stencil_raw;
		precond.conv_kernel=kernel_raw;
		BuildConvSAIDirectMap();
		BuildConvSAIDirectNeighbors();
		if (IFROOT)
			printf("ConvSAI preconditioner loaded: %zu stencil, direct sparse convolution, grid %zux%zux%zu\n",
				n_stencil,gx,gy,gz);
		return;
	}

	// Allocate frequency-domain kernel: 9 components × gridN
	precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"convsai Phat");

	// Build spatial kernel and FFT to get Phat
	// For each (a,b) pair (9 total): place kernel blocks on grid, then FFT
	{
		doublecomplex *spatial=(doublecomplex *)voidVector(gridN*sizeof(doublecomplex),ALL_POS,"convsai spatial");
		void *build_plan=(void *)fftw_plan_dft_3d(
			(int)gz,(int)gy,(int)gx,
			(fftw_complex *)spatial,
			(fftw_complex *)spatial,
			FFTW_FORWARD,FFTW_MEASURE);

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
			fftw_execute((fftw_plan)build_plan);

			// Store in Phat: component (a,b) at offset (3*a+b)*gridN
			memcpy(precond.conv_Phat+(3*a+b)*gridN,spatial,gridN*sizeof(doublecomplex));
		}

		fftw_destroy_plan((fftw_plan)build_plan);
		free(spatial);
	}

	free(stencil_raw);
	free(kernel_raw);

	if (!ShouldUseConvSAIDistributed()) {
		// Allocate work buffers and plans for the replicated FFT fallback.
		precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"convsai work_in");
		precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"convsai work_out");
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
	}

	if (IFROOT)
		printf("ConvSAI preconditioner loaded: %zu stencil, FFT grid %zux%zux%zu%s\n",
			n_stencil,gx,gy,gz,ShouldUseConvSAIDistributed() ? ", distributed MPI apply" : "");
#endif // FFTW3
}

//======================================================================================================================

#if defined(PARALLEL) && !defined(SPARSE)
static inline size_t PrecondIndexSliceYZ(const size_t y,const size_t z)
{
	return y*gridZ+z;
}

static inline size_t PrecondIndexSliceZY(const size_t y,const size_t z)
{
	return z*gridY+y;
}

static inline size_t PrecondIndexGarbledX(const size_t x,const size_t y,const size_t z)
{
	return ((z%local_Nz)*smallY+y)*gridX+(z/local_Nz)*local_Nx+x%local_Nx;
}

static inline size_t PrecondIndexXmatrix(const size_t x,const size_t y,const size_t z)
{
	return (z*smallY+y)*gridX+x;
}

static void ApplyConvSAIDistributedFFT(const doublecomplex *in,doublecomplex *out,size_t n)
/* MPI ConvSAI apply using ADDA's distributed FFT layout.
 *
 * The old MPI path gathered a full grid on every rank, then every rank performed
 * the complete 3D FFT and dense 3x3 frequency multiply. This function follows
 * MatVec's decomposition instead: z-slab input, FFT-X, BlockTranspose to x-slabs,
 * local Y/Z FFTs and local frequency multiply, then the inverse path.
 */
{
	size_t i,x,y,z,comp,a,b,k;
	size_t ndip=n/3;
	size_t gridN=precond.conv_gridN;
	double inv_gridN=1.0/(double)gridN;

	for (i=0;i<3*local_Nsmall;i++) Xmatrix[i]=0.0;

	for (i=0;i<ndip;i++) {
		size_t j=3*i;
		size_t idx=PrecondIndexXmatrix(position[j],position[j+1],position[j+2]);
		for (comp=0;comp<3;comp++)
			Xmatrix[idx+comp*local_Nsmall]=in[j+comp];
	}

	fftX(FFT_FORWARD);
	BlockTranspose(Xmatrix,NULL);

	for (x=local_x0;x<local_x1;x++) {
		for (i=0;i<3*gridYZ;i++) slices[i]=0.0;

		for (y=0;y<(size_t)boxY;y++) for (z=0;z<(size_t)boxZ;z++) {
			size_t src=PrecondIndexGarbledX(x,y,z);
			size_t dst=PrecondIndexSliceYZ(y,z);
			for (comp=0;comp<3;comp++)
				slices[dst+comp*gridYZ]=Xmatrix[src+comp*local_Nsmall];
		}

		fftZ(FFT_FORWARD);
		TransposeYZ(FFT_FORWARD);
		fftY(FFT_FORWARD);

		for (z=0;z<gridZ;z++) for (y=0;y<gridY;y++) {
			size_t yz=PrecondIndexSliceZY(y,z);
			size_t xyz=(z*gridY+y)*gridX+x;
			doublecomplex in0=slices_tr[yz+0*gridYZ];
			doublecomplex in1=slices_tr[yz+1*gridYZ];
			doublecomplex in2=slices_tr[yz+2*gridYZ];
			doublecomplex outv[3];

			for (a=0;a<3;a++) {
				k=(3*a+0)*gridN+xyz;
				outv[a]=precond.conv_Phat[k]*in0;
				k=(3*a+1)*gridN+xyz;
				outv[a]+=precond.conv_Phat[k]*in1;
				k=(3*a+2)*gridN+xyz;
				outv[a]+=precond.conv_Phat[k]*in2;
			}
			for (b=0;b<3;b++)
				slices_tr[yz+b*gridYZ]=outv[b];
		}

		fftY(FFT_BACKWARD);
		TransposeYZ(FFT_BACKWARD);
		fftZ(FFT_BACKWARD);

		for (y=0;y<(size_t)boxY;y++) for (z=0;z<(size_t)boxZ;z++) {
			size_t src=PrecondIndexSliceYZ(y,z);
			size_t dst=PrecondIndexGarbledX(x,y,z);
			for (comp=0;comp<3;comp++)
				Xmatrix[dst+comp*local_Nsmall]=slices[src+comp*gridYZ];
		}
	}

	BlockTranspose(Xmatrix,NULL);
	fftX(FFT_BACKWARD);

	for (i=0;i<ndip;i++) {
		size_t j=3*i;
		size_t idx=PrecondIndexXmatrix(position[j],position[j+1],position[j+2]);
		for (comp=0;comp<3;comp++)
			out[j+comp]=Xmatrix[idx+comp*local_Nsmall]*inv_gridN;
	}
}
#endif

//======================================================================================================================

static void ApplyConvSAIDirect(const doublecomplex *in,doublecomplex *out,size_t n)
/* Apply ConvSAI by direct sparse convolution over the stored stencil.
 * This path is intended for threshold-pruned kernels where n_stencil is small
 * enough that O(N*n_stencil) beats redundant full-grid FFTs in MPI mode.
 */
{
	const doublecomplex *in_full;
	size_t gx=precond.conv_gx;
	size_t gy=precond.conv_gy;
	size_t gz=precond.conv_gz;
	const int32_t *stencil=precond.conv_stencil;
	const doublecomplex *kernel=precond.conv_kernel;
	const int64_t *grid_to_dipole=precond.conv_grid_to_dipole;
	size_t ndip=n/3;
	size_t i,s;

#ifdef PARALLEL
	AllGather((void *)in,precond.gather_buf,cmplx3_type,NULL);
	in_full=precond.gather_buf;
#else
	in_full=in;
#endif

	for (i=0;i<ndip;i++) {
		int px=position[3*i+0];
		int py=position[3*i+1];
		int pz=position[3*i+2];
		doublecomplex out0=0,out1=0,out2=0;
#ifdef PARALLEL
		pz+=local_z0;
#endif
		if (precond.conv_direct_row_ptr!=NULL) {
			size_t e;
			for (e=precond.conv_direct_row_ptr[i];e<precond.conv_direct_row_ptr[i+1];e++) {
				const doublecomplex *x=in_full+3*(size_t)precond.conv_direct_dipole[e];
				const doublecomplex *K=kernel+9*(size_t)precond.conv_direct_stencil[e];
				out0+=K[0]*x[0]+K[1]*x[1]+K[2]*x[2];
				out1+=K[3]*x[0]+K[4]*x[1]+K[5]*x[2];
				out2+=K[6]*x[0]+K[7]*x[1]+K[8]*x[2];
			}
		}
		else for (s=0;s<precond.conv_n_stencil;s++) {
			int64_t dip=ConvSAIDirectNeighborAt(px,py,pz,s);
			if (dip>=0) {
				const doublecomplex *x=in_full+3*(size_t)dip;
				const doublecomplex *K=kernel+9*s;
				out0+=K[0]*x[0]+K[1]*x[1]+K[2]*x[2];
				out1+=K[3]*x[0]+K[4]*x[1]+K[5]*x[2];
				out2+=K[6]*x[0]+K[7]*x[1]+K[8]*x[2];
			}
		}
		out[3*i+0]=out0;
		out[3*i+1]=out1;
		out[3*i+2]=out2;
	}
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
	if (precond.conv_direct) {
		ApplyConvSAIDirect(in,out,n);
		return;
	}

#if defined(PARALLEL) && !defined(SPARSE)
	if (ShouldUseConvSAIDistributed()) {
		ApplyConvSAIDistributedFFT(in,out,n);
		return;
	}
#endif

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
 * CONVSAI/FFT: By default uses ADDA's distributed FFT decomposition. Set
 *              ADDA_CONVSAI_DISTRIBUTED=0 to use the old replicated full-grid FFT fallback.
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
			if (precond.conv_plan_fwd!=NULL)
				fftw_destroy_plan((fftw_plan)precond.conv_plan_fwd);
			if (precond.conv_plan_bwd!=NULL)
				fftw_destroy_plan((fftw_plan)precond.conv_plan_bwd);
#endif
			free(precond.conv_stencil);
			free(precond.conv_kernel);
			free(precond.conv_grid_to_dipole);
			free(precond.conv_direct_row_ptr);
			free(precond.conv_direct_dipole);
			free(precond.conv_direct_stencil);
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
