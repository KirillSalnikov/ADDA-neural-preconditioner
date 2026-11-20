/* Neural preconditioner: load binary .precond file and apply SAI, ILU, POLY, CONVSAI, or FFTDIRECT.
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
 *   mode=4 (FFTDIRECT): nnz = 0
 *                   Data = dims[3](u64), Phat[9*gridN*2](f64), component-major
 *   mode=5 (FFTDIRECT_XSLAB_F32): nnz = 0
 *                   Data = dims[3](u64), Phat[gridX*gridZ*gridY*9*2](f32), x-slab-major
 *
 * Copyright (C) ADDA contributors
 * This file is part of ADDA.
 */
#include "const.h" // keep this first
#include "precond.h"
#include "comm.h"
#include "fft.h"   // defines FFTW3 macro when FFTW3 is available
#include "cmplx.h" // REIM3V for -precond_npw_match (plan task D2m)
#include "interaction.h" // InterTerm_int for -precond_npw_match (plan task D2m)
#include "io.h"
#include "memory.h"
#include "timing.h"
#include "vars.h"
// system headers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
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
extern TIME_TYPE Timing_Precond,Timing_OneIterPrecond;

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

static void PackConvSAILocalPhat(void)
/* Pack the full frequency-domain kernel into a local x-slab layout for MPI
 * distributed apply. The original layout is full-grid contiguous in x and
 * component-major. The packed layout stores the 9 matrix entries next to each
 * local frequency point, which keeps the hot yz loop contiguous and lets each
 * rank free unused x slabs.
 */
{
#if defined(PARALLEL) && !defined(SPARSE)
	size_t comp,x,y,z,lx,yz,xyz;
	size_t local_grid;
	doublecomplex *packed;

	if (!ShouldUseConvSAIDistributed()) return;
	if (precond.conv_Phat==NULL) return;

	local_grid=local_Nx*gridYZ;
	packed=(doublecomplex *)voidVector(9*local_grid*sizeof(doublecomplex),ALL_POS,
		"convsai local Phat");

	for (x=local_x0;x<local_x1;x++) {
		lx=x-local_x0;
		for (z=0;z<gridZ;z++) for (y=0;y<gridY;y++) {
			yz=z*gridY+y;
			xyz=(z*gridY+y)*gridX+x;
			for (comp=0;comp<9;comp++)
				packed[9*(lx*gridYZ+yz)+comp]=precond.conv_Phat[comp*precond.conv_gridN+xyz];
		}
	}

	free(precond.conv_Phat);
	precond.conv_Phat=NULL;
	precond.conv_Phat_local=packed;
#endif
}

//======================================================================================================================

static void LoadFFTDirectLocalPhat(FILE *f,const char *filename,size_t gridN)
/* Read only this MPI rank's local x-slab from a mode=4 FFTDIRECT file.
 *
 * FFTDIRECT stores 9 component-major full grids as interleaved doubles in
 * z,y,x order. In distributed MPI apply each rank only needs local_x0:local_x1,
 * so reading the whole Phat on every rank creates large redundant import I/O.
 */
{
#if defined(PARALLEL) && !defined(SPARSE)
	const size_t header_bytes=8*(5+3);
	const size_t complex_bytes=2*sizeof(double);
	const size_t row_count=local_Nx;
	size_t comp,y,z,lx,yz;
	double *raw;
	doublecomplex *packed;

	packed=(doublecomplex *)voidVector(9*local_Nx*gridYZ*sizeof(doublecomplex),ALL_POS,
		"fftdirect local Phat");
	raw=(double *)voidVector(2*row_count*sizeof(double),ALL_POS,"fftdirect local Phat row");

	for (comp=0;comp<9;comp++) {
		const off_t comp_offset=(off_t)header_bytes+(off_t)(comp*gridN*complex_bytes);
		for (z=0;z<gridZ;z++) for (y=0;y<gridY;y++) {
			const size_t row0=(z*gridY+y)*gridX+local_x0;
			const off_t offset=comp_offset+(off_t)(row0*complex_bytes);
			if (fseeko(f,offset,SEEK_SET)!=0)
				LogError(ONE_POS,"Failed to seek local FFTDIRECT Phat in preconditioner file '%s'",filename);
			if (fread(raw,sizeof(double),2*row_count,f)!=2*row_count)
				LogError(ONE_POS,"Failed to read local FFTDIRECT Phat from preconditioner file '%s'",filename);
			yz=z*gridY+y;
			for (lx=0;lx<local_Nx;lx++)
				packed[9*(lx*gridYZ+yz)+comp]=raw[2*lx]+I*raw[2*lx+1];
		}
	}

	free(raw);
	precond.conv_Phat_local=packed;
#else
	(void)f;
	(void)filename;
	(void)gridN;
#endif
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
	precond.conv_Phat_local=NULL;
	precond.conv_work_in=NULL;
	precond.conv_work_out=NULL;
	precond.conv_plan_fwd=NULL;
	precond.conv_plan_bwd=NULL;

	if (ShouldUseConvSAIDistributed()) {
		LoadFFTDirectLocalPhat(f,filename,gridN);
	} else {
		/* Allocate Phat and read directly for replicated/sequential apply. */
		precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"fftdirect Phat");
		ReadComplexValues(f,precond.conv_Phat,9*gridN,filename,"FFTDIRECT Phat");

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

static void LoadFFTDirectXSlabF32LocalPhat(FILE *f,const char *filename)
/* Read this MPI rank's local x-slab from a mode=5 FFTDIRECT_XSLAB_F32 file.
 *
 * Mode=5 stores complex float32 in x-slab-major order:
 *   x, z, y, component(0..8), re/im
 * This matches ADDA's distributed FFT x-slab ownership, so one rank reads one
 * contiguous block instead of many row-sized seeks.
 */
{
#if defined(PARALLEL) && !defined(SPARSE)
	const size_t header_bytes=8*(5+3);
	const size_t complex_bytes=2*sizeof(float);
	const size_t total_complex=local_Nx*gridYZ*9;
	const off_t offset=(off_t)header_bytes+
		(off_t)(local_x0*gridYZ*9*complex_bytes);
	size_t lx,yz,comp,raw_i;
	float *raw;
	doublecomplex *packed;

	packed=(doublecomplex *)voidVector(total_complex*sizeof(doublecomplex),ALL_POS,
		"fftdirect xslab f32 local Phat");
	raw=(float *)voidVector(2*total_complex*sizeof(float),ALL_POS,
		"fftdirect xslab f32 raw");

	if (fseeko(f,offset,SEEK_SET)!=0)
		LogError(ONE_POS,"Failed to seek local FFTDIRECT_XSLAB_F32 Phat in preconditioner file '%s'",filename);
	if (fread(raw,sizeof(float),2*total_complex,f)!=2*total_complex)
		LogError(ONE_POS,"Failed to read local FFTDIRECT_XSLAB_F32 Phat from preconditioner file '%s'",filename);

	for (lx=0;lx<local_Nx;lx++) for (yz=0;yz<gridYZ;yz++) for (comp=0;comp<9;comp++) {
		raw_i=2*((lx*gridYZ+yz)*9+comp);
		packed[9*(lx*gridYZ+yz)+comp]=(double)raw[raw_i]+I*(double)raw[raw_i+1];
	}

	free(raw);
	precond.conv_Phat_local=packed;
#else
	(void)f;
	(void)filename;
#endif
}

//======================================================================================================================

static void LoadFFTDirectXSlabF32FullPhat(FILE *f,const char *filename,size_t gridN)
/* Load a mode=5 FFTDIRECT_XSLAB_F32 file for sequential or replicated MPI fallback.
 * The in-memory layout is converted back to the component-major Phat layout used
 * by the existing replicated FFT apply path.
 */
{
	size_t x,z,y,yz,comp,idx,raw_i;
	const size_t slab_complex=gridYZ*9;
	float *raw;

	precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,
		"fftdirect xslab f32 Phat");
	raw=(float *)voidVector(2*slab_complex*sizeof(float),ALL_POS,
		"fftdirect xslab f32 slab");

	for (x=0;x<gridX;x++) {
		if (fread(raw,sizeof(float),2*slab_complex,f)!=2*slab_complex)
			LogError(ONE_POS,"Failed to read FFTDIRECT_XSLAB_F32 slab from preconditioner file '%s'",filename);
		for (z=0;z<gridZ;z++) for (y=0;y<gridY;y++) {
			yz=z*gridY+y;
			idx=yz*gridX+x;
			for (comp=0;comp<9;comp++) {
				raw_i=2*(yz*9+comp);
				precond.conv_Phat[comp*gridN+idx]=(double)raw[raw_i]+I*(double)raw[raw_i+1];
			}
		}
	}

	free(raw);
}

//======================================================================================================================

static void LoadFFTDirectXSlabF32(FILE *f,const char *filename)
/* Load FFT-direct preconditioner stored as x-slab-major complex float32.
 * Computation still uses ADDA's doublecomplex buffers; quantization affects
 * only storage/import bandwidth, not solver arithmetic.
 */
{
#ifndef FFTW3
	LogError(ONE_POS,"FFTDIRECT_XSLAB_F32 preconditioner requires FFTW3");
#else
	uint64_t dims[3];
	size_t gx,gy,gz,gridN;

	if (fread(dims,sizeof(uint64_t),3,f)!=3)
		LogError(ONE_POS,"Failed to read grid dims from preconditioner file '%s'",filename);
	gx=(size_t)dims[0];
	gy=(size_t)dims[1];
	gz=(size_t)dims[2];
	gridN=gx*gy*gz;

	if (gx!=gridX || gy!=gridY || gz!=gridZ)
		LogError(ONE_POS,"FFTDIRECT_XSLAB_F32 grid %zux%zux%zu != ADDA grid %zux%zux%zu in '%s'",
			gx,gy,gz,gridX,gridY,gridZ,filename);

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
	precond.conv_Phat_local=NULL;
	precond.conv_work_in=NULL;
	precond.conv_work_out=NULL;
	precond.conv_plan_fwd=NULL;
	precond.conv_plan_bwd=NULL;

	if (ShouldUseConvSAIDistributed()) {
		LoadFFTDirectXSlabF32LocalPhat(f,filename);
	} else {
		LoadFFTDirectXSlabF32FullPhat(f,filename,gridN);
		precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,
			"fftdirect xslab f32 work_in");
		precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,
			"fftdirect xslab f32 work_out");
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
		printf("FFTDIRECT_XSLAB_F32 preconditioner loaded: grid %zux%zux%zu, %zu Phat values%s\n",
			gx,gy,gz,9*gridN,ShouldUseConvSAIDistributed() ? ", distributed MPI apply" : "");
#endif
}

//======================================================================================================================

static bool IsConvFFTMode(void)
{
	return precond.mode==PRECOND_MODE_CONVSAI || precond.mode==PRECOND_MODE_FFTDIRECT ||
		precond.mode==PRECOND_MODE_FFTDIRECT_XSLAB_F32;
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
	precond.conv_Phat_local=NULL;
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

	if (ShouldUseConvSAIDistributed()) {
		PackConvSAILocalPhat();
	} else {
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

static inline size_t PrecondIndexDmatrix(const size_t x,const size_t y_in,const size_t z_in)
{
	size_t y=y_in;
	size_t z=z_in;
	if (y>=DsizeY) y=gridY-y;
	if (z>=DsizeZ) z=gridZ-z;
	return NDCOMP*((x*DsizeZ+z)*DsizeY+y);
}

bool PrecondCanApplyMatVecFused(void)
{
	bool forced=false;
	const char *env=getenv("ADDA_PRECOND_FUSED_MATVEC");
	if (!ParseBoolEnv(env,&forced) || !forced) return false;
	if (!use_precond) return false;
	if (!IsConvFFTMode()) return false;
	if (precond.conv_direct) return false;
	if (!ShouldUseConvSAIDistributed()) return false;
	if (surface) return false;
	if (Nmat!=1) return false;
	return true;
}

void PrecondApplyMatVecFused(const doublecomplex *in,doublecomplex *out,size_t n)
/* Experimental approximate fused operator: out ~= M*A*in.
 *
 * This saves one full FFT round by applying M_hat to A_hat before returning to
 * particle space. It intentionally skips the particle-mask projection between
 * A and M, so it is only enabled through ADDA_PRECOND_FUSED_MATVEC=1.
 */
{
	size_t i,x,y,z,comp;
	size_t ndip=n/3;
	size_t gridN=precond.conv_gridN;
	const bool local_Phat=(precond.conv_Phat_local!=NULL);
	const doublecomplex *Phat=local_Phat ? precond.conv_Phat_local : precond.conv_Phat;
	const size_t PhatN=local_Phat ? local_Nx*gridYZ : gridN;
	const doublecomplex s0=cc_sqrt[0][0];
	const doublecomplex s1=cc_sqrt[0][1];
	const doublecomplex s2=cc_sqrt[0][2];
	double inv_gridN=1.0/(double)gridN;
	TIME_TYPE tstart=GET_TIME();

	if (!PrecondCanApplyMatVecFused())
		LogError(ONE_POS,"ADDA_PRECOND_FUSED_MATVEC is incompatible with this run");

	memset(Xmatrix,0,3*local_Nsmall*sizeof(doublecomplex));

	for (i=0;i<ndip;i++) {
		size_t j=3*i;
		size_t idx=PrecondIndexXmatrix(position[j],position[j+1],position[j+2]);
		Xmatrix[idx+0*local_Nsmall]=s0*in[j+0];
		Xmatrix[idx+1*local_Nsmall]=s1*in[j+1];
		Xmatrix[idx+2*local_Nsmall]=s2*in[j+2];
	}

	fftX(FFT_FORWARD);
	BlockTranspose(Xmatrix,NULL);

	for (x=local_x0;x<local_x1;x++) {
		memset(slices,0,3*gridYZ*sizeof(doublecomplex));

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
			size_t pidx=local_Phat ? (x-local_x0)*gridYZ+yz : xyz;
			const doublecomplex *P=local_Phat ? Phat+9*pidx : NULL;
			size_t didx=PrecondIndexDmatrix(x-local_x0,y,z);
			doublecomplex f0=Dmatrix[didx+0];
			doublecomplex f1=Dmatrix[didx+1];
			doublecomplex f2=Dmatrix[didx+2];
			doublecomplex f3=Dmatrix[didx+3];
			doublecomplex f4=Dmatrix[didx+4];
			doublecomplex f5=Dmatrix[didx+5];
			doublecomplex x0=slices_tr[yz+0*gridYZ];
			doublecomplex x1=slices_tr[yz+1*gridYZ];
			doublecomplex x2=slices_tr[yz+2*gridYZ];
			doublecomplex d0,d1,d2,a0,a1,a2;

			if (reduced_FFT) {
				if (y>=DsizeY) {
					f1*=-1;
					if (z>=DsizeZ) f2*=-1;
					else f4*=-1;
				}
				else if (z>=DsizeZ) {
					f2*=-1;
					f4*=-1;
				}
			}

			d0=f0*x0+f1*x1+f2*x2;
			d1=f1*x0+f3*x1+f4*x2;
			d2=f2*x0+f4*x1+f5*x2;

			a0=x0/s0+s0*d0;
			a1=x1/s1+s1*d1;
			a2=x2/s2+s2*d2;

			if (local_Phat) {
				slices_tr[yz+0*gridYZ]=P[0]*a0+P[1]*a1+P[2]*a2;
				slices_tr[yz+1*gridYZ]=P[3]*a0+P[4]*a1+P[5]*a2;
				slices_tr[yz+2*gridYZ]=P[6]*a0+P[7]*a1+P[8]*a2;
			}
			else {
				slices_tr[yz+0*gridYZ]=Phat[0*PhatN+pidx]*a0+
					Phat[1*PhatN+pidx]*a1+Phat[2*PhatN+pidx]*a2;
				slices_tr[yz+1*gridYZ]=Phat[3*PhatN+pidx]*a0+
					Phat[4*PhatN+pidx]*a1+Phat[5*PhatN+pidx]*a2;
				slices_tr[yz+2*gridYZ]=Phat[6*PhatN+pidx]*a0+
					Phat[7*PhatN+pidx]*a1+Phat[8*PhatN+pidx]*a2;
			}
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

	{
		TIME_TYPE elapsed=GET_TIME()-tstart;
		Timing_Precond+=elapsed;
		Timing_OneIterPrecond+=elapsed;
	}
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
	size_t i,x,y,z,comp;
	size_t ndip=n/3;
	size_t gridN=precond.conv_gridN;
	const bool local_Phat=(precond.conv_Phat_local!=NULL);
	const doublecomplex *Phat=local_Phat ? precond.conv_Phat_local : precond.conv_Phat;
	const size_t PhatN=local_Phat ? local_Nx*gridYZ : gridN;
	double inv_gridN=1.0/(double)gridN;

	memset(Xmatrix,0,3*local_Nsmall*sizeof(doublecomplex));

	for (i=0;i<ndip;i++) {
		size_t j=3*i;
		size_t idx=PrecondIndexXmatrix(position[j],position[j+1],position[j+2]);
		for (comp=0;comp<3;comp++)
			Xmatrix[idx+comp*local_Nsmall]=in[j+comp];
	}

	fftX(FFT_FORWARD);
	BlockTranspose(Xmatrix,NULL);

	for (x=local_x0;x<local_x1;x++) {
		memset(slices,0,3*gridYZ*sizeof(doublecomplex));

		for (y=0;y<(size_t)boxY;y++) for (z=0;z<(size_t)boxZ;z++) {
			size_t src=PrecondIndexGarbledX(x,y,z);
			size_t dst=PrecondIndexSliceYZ(y,z);
			for (comp=0;comp<3;comp++)
				slices[dst+comp*gridYZ]=Xmatrix[src+comp*local_Nsmall];
		}

		fftZ(FFT_FORWARD);
		TransposeYZ(FFT_FORWARD);
		fftY(FFT_FORWARD);

		if (local_Phat) {
			const size_t px0=(x-local_x0)*gridYZ;
			for (z=0;z<gridZ;z++) for (y=0;y<gridY;y++) {
				size_t yz=PrecondIndexSliceZY(y,z);
				const doublecomplex *P=Phat+9*(px0+yz);
				doublecomplex in0=slices_tr[yz+0*gridYZ];
				doublecomplex in1=slices_tr[yz+1*gridYZ];
				doublecomplex in2=slices_tr[yz+2*gridYZ];

				slices_tr[yz+0*gridYZ]=P[0]*in0+P[1]*in1+P[2]*in2;
				slices_tr[yz+1*gridYZ]=P[3]*in0+P[4]*in1+P[5]*in2;
				slices_tr[yz+2*gridYZ]=P[6]*in0+P[7]*in1+P[8]*in2;
			}
		}
		else {
			for (z=0;z<gridZ;z++) for (y=0;y<gridY;y++) {
				size_t yz=PrecondIndexSliceZY(y,z);
				size_t xyz=(z*gridY+y)*gridX+x;
				doublecomplex in0=slices_tr[yz+0*gridYZ];
				doublecomplex in1=slices_tr[yz+1*gridYZ];
				doublecomplex in2=slices_tr[yz+2*gridYZ];

				slices_tr[yz+0*gridYZ]=Phat[0*PhatN+xyz]*in0+
					Phat[1*PhatN+xyz]*in1+Phat[2*PhatN+xyz]*in2;
				slices_tr[yz+1*gridYZ]=Phat[3*PhatN+xyz]*in0+
					Phat[4*PhatN+xyz]*in1+Phat[5*PhatN+xyz]*in2;
				slices_tr[yz+2*gridYZ]=Phat[6*PhatN+xyz]*in0+
					Phat[7*PhatN+xyz]*in1+Phat[8*PhatN+xyz]*in2;
			}
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

#if !defined(PARALLEL) || defined(SPARSE)
bool PrecondCanApplyMatVecFused(void)
{
	return false;
}

void PrecondApplyMatVecFused(const doublecomplex *in,doublecomplex *out,size_t n)
{
	(void)in;
	(void)out;
	(void)n;
	LogError(ONE_POS,"ADDA_PRECOND_FUSED_MATVEC requires non-sparse MPI mode");
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
	const size_t actual_n=3*nvoid_Ndip;

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
	} else if (precond.mode==PRECOND_MODE_FFTDIRECT_XSLAB_F32) {
		LoadFFTDirectXSlabF32(f,filename);
	} else {
		LogError(ONE_POS,"Unknown preconditioner mode %lu in file '%s'",(unsigned long)precond.mode,filename);
	}

	if (IsConvFFTMode()) {
		/* Convolution kernels (modes 3-5) do not depend on n, but the header n is 3 x the number of dipoles of the
		 * particle the kernel was made for. A different n means a different particle (audit finding C4: the
		 * network's Python hex prism had 21,504 dipoles where ADDA has 28,416), so it is an error (plan task B1).
		 * n = 0 means "unknown" (exporters without the geometry write it) and is accepted with a note.
		 * ADDA_PRECOND_ALLOW_N_MISMATCH=1 accepts a mismatch with a warning, to rerun old (legacy) files.
		 */
		if (precond.n==0) {
			if (IFROOT) printf("Conv/FFT preconditioner header n=0 (unknown particle), using ADDA particle n=%zu\n",
				actual_n);
		}
		else if (precond.n!=actual_n) {
			const char *allow=getenv("ADDA_PRECOND_ALLOW_N_MISMATCH");
			if (allow!=NULL && atoi(allow)!=0) {
				if (IFROOT) printf("WARNING: Conv/FFT preconditioner header n=%zu (%zu dipoles) != ADDA particle "
					"n=%zu (%zu dipoles); used anyway because ADDA_PRECOND_ALLOW_N_MISMATCH is set\n",
					precond.n,precond.n/3,actual_n,actual_n/3);
			}
			else LogError(ONE_POS,"Preconditioner '%s' was made for a different particle: header n=%zu (%zu dipoles), "
				"ADDA particle n=%zu (%zu dipoles). Export it with ADDA's geometry (e.g. --geom of the exporters), "
				"or set ADDA_PRECOND_ALLOW_N_MISMATCH=1 to use it anyway",filename,precond.n,precond.n/3,
				actual_n,actual_n/3);
		}
		precond.n=actual_n;
	}
	else if (precond.n!=actual_n) {
		LogError(ONE_POS,"Preconditioner vector length %zu != ADDA particle vector length %zu in '%s'",
			precond.n,actual_n,filename);
	}

	// Allocate temporary buffers (needed by all modes for left preconditioning in iterative.c)
	precond.tmp=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp");
	precond.tmp2=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp2");
	precond.r_actual=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond r_actual");
	precond.gather_buf=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond gather_buf");

	fclose(f);
	/* Only FGMRES (right preconditioning), BiCGStab (left) and BiCG (preconditioned COCG, z=M.r, plan task D2a) apply
	 * the preconditioner in every iteration; the other solvers (qmr, qmr2, bcgs2, cgnr, csym) never call PrecondApply,
	 * so a run with them would silently be unpreconditioned (audit finding M1, plan task B3). -precond_init_only works
	 * with every solver.
	 */
	if (!precond_init_only && IterMethod!=IT_FGMRES && IterMethod!=IT_BICGSTAB && IterMethod!=IT_BICG_CS)
		LogError(ONE_POS,"-precond is applied only by -iter fgmres, -iter bicgstab and -iter bicg (or once, with "
			"-precond_init_only); the chosen iterative solver would ignore '%s'",filename);
	use_precond=true;
}

//======================================================================================================================
/* Windowed medium kernel ("npw") built inside ADDA: option -precond_npw <s> [tri|hann] (plan task D2f, stage D design
 * 2 stage 2). It is a pure addition: -precond and its file loaders above are not changed, and the kernel is put into
 * exactly the arrays the mode-4 (FFTDIRECT) loader fills, so the same apply code runs as with a mode-4 file.
 *
 * Math (units of the dipole size d, so k = kd), the same as apps/export_circulant_precond.py --kernel npw:
 *   1. Generator on the FFT grid j in [0,L)^3 (L = gridX, gridY, gridZ; L_i >= 2 box_i):
 *        g(j) = sum over e in {0,1}^3 of w((j - e L)/L) G(j - e L),
 *      with the separable window w(t) = h(t_x) h(t_y) h(t_z) (tri: h(u) = 1 - |u|, hann: (1 + cos(pi u))/2)
 *      and the point-dipole Green's tensor G (G(0) = 0).
 *   2. Symbol W(q) = forward FFT of g (one symmetric 3x3 block per frequency).
 *   3. P(q) = (I - c_box W(q))^-1, where c_box is the LDR polarizability (S = 0) of the damped material with
 *      contrast eps (1 + i s), eps = m^2 of the particle.
 * Each MPI rank computes only the frequencies of its own x-slab (local_x0 .. local_x1), so no communication is
 * needed. The generator has a mirror symmetry, g(L - j) = sign g(j) per axis (off-diagonal components that contain
 * the mirrored axis change sign), so only the points with j_i <= L_i/2 are evaluated (1/8 of the grid).
 *
 * ADDA_PRECOND_NPW_COMPARE=<mode-4 file> compares the built kernel with a file made by the Python exporter (for the
 * same particle, m, dpl, s, window and number of MPI ranks) and prints the largest and the relative L2 difference.
 */
#ifdef FFTW3

// Row and column of each of the 6 components of the symmetric storage xx, xy, xz, yy, yz, zz (ADDA's Dmatrix order)
static const int npw_sym_row[6]={0,0,0,1,1,2};
static const int npw_sym_col[6]={0,1,2,1,2,2};

static double npw_window_shape=0.0;
static double npw_scale=1.0, npw_shift=0.0;

static double NPWWindow(const double u_in,const int window)
/* 1-D window h(u) for u = |t|/L: h(0) = 1 and h(u) = 0 for u >= 1.
 * tri: 1 - u (gives T. Chan's circulant of period L); hann: (1 + cos(pi u))/2.
 */
{
	const double u=fabs(u_in);
	if (u>=1.0) return 0.0;
	if (window==PRECOND_NPW_WINDOW_HANN) {
		const double h=0.5*(1.0+cos(PI*u));
		/* u and 1-u are image partners. This cubic residual is antisymmetric
		 * about 1/2, preserving their partition of unity. */
		return h+npw_window_shape*h*(1.0-h)*(2.0*h-1.0);
	}
	return 1.0-u;
}

static void NPWGreen(const double tx,const double ty,const double tz,const double kd_npw,doublecomplex g[6])
/* Point-dipole Green's tensor G(t) in the symmetric storage, G(0) = 0. Formula of ADDA's InterTerm_core and of
 * green_tensor_at in the Python exporter:
 *   G(r) = exp(ikr)/r^3 [ (k^2 r^2 - 1 + ikr) I + (3 - k^2 r^2 - 3ikr) rhat rhat ].
 */
{
	const double r2=tx*tx+ty*ty+tz*tz;
	double r,kr,unit[3];
	doublecomplex expval,diag_coef,outer_coef;
	int comp;

	if (r2==0) {
		for (comp=0;comp<6;comp++) g[comp]=0;
		return;
	}
	r=sqrt(r2);
	kr=kd_npw*r;
	expval=cexp(I*kr)/(r2*r);
	diag_coef=(kr*kr-1.0+I*kr)*expval;   // multiplies the identity
	outer_coef=(3.0-kr*kr-3.0*I*kr)*expval; // multiplies rhat rhat^T
	unit[0]=tx/r;
	unit[1]=ty/r;
	unit[2]=tz/r;
	for (comp=0;comp<6;comp++) {
		g[comp]=outer_coef*unit[npw_sym_row[comp]]*unit[npw_sym_col[comp]];
		if (npw_sym_row[comp]==npw_sym_col[comp]) g[comp]+=diag_coef;
	}
}

static void NPWGenerator(const size_t j[3],const size_t L[3],const double kd_npw,const int window,doublecomplex g[6])
/* Generator g(j) of step 1 at one grid point j (0 <= j_i < L_i): the windowed sum over the 8 images j - e L. The
 * images are visited in the order of the Python exporter (e_x outer, e_z inner).
 */
{
	double t[3][2],h[3][2];
	doublecomplex term[6];
	int axis,ex,ey,ez,comp;

	for (axis=0;axis<3;axis++) {
		t[axis][0]=(double)j[axis];
		t[axis][1]=(double)j[axis]-(double)L[axis];
		h[axis][0]=NPWWindow(t[axis][0]/(double)L[axis],window);
		h[axis][1]=NPWWindow(t[axis][1]/(double)L[axis],window);
	}
	for (comp=0;comp<6;comp++) g[comp]=0;
	for (ex=0;ex<2;ex++) for (ey=0;ey<2;ey++) for (ez=0;ez<2;ez++) {
		const double weight=h[0][ex]*h[1][ey]*h[2][ez];
		if (weight==0) continue; // this image carries no weight here
		NPWGreen(t[0][ex],t[1][ey],t[2][ez],kd_npw,term);
		for (comp=0;comp<6;comp++) g[comp]+=weight*term[comp];
	}
}

static double NPWMirrorSign(const int comp,const int axis)
/* Sign of component comp of g when the coordinate along axis is mirrored (j -> L - j): -1 for an off-diagonal
 * component that contains this axis, +1 otherwise.
 */
{
	const int row=npw_sym_row[comp],col=npw_sym_col[comp];
	return (row!=col && (row==axis || col==axis)) ? -1.0 : 1.0;
}

static doublecomplex NPWBoxPolarizability(const double damping)
/* LDR polarizability (S = 0, dipole volume 1) of the damped material: contrast eps (1 + i s), eps = m^2.
 * Same as polarizability(damped_index(m, s, "eps"), kd) in the Python exporter.
 */
{
	const doublecomplex eps_box=ref_index[0]*ref_index[0]*(1.0+I*damping);
	const doublecomplex alpha_cm=THREE_OVER_FOUR_PI*(eps_box-1.0)/(eps_box+2.0);
	const doublecomplex m_term=(LDR_B1+LDR_B2*eps_box)*kd*kd+2.0*I*kd*kd*kd/3.0;
	return alpha_cm/(1.0-alpha_cm*m_term);
}

static doublecomplex *NPWBuildSlab(const size_t x0,const size_t nx,const double damping,const int window,
	double *max_entry)
/* P(q) for the frequencies q_x in [x0, x0 + nx) and all q_y, q_z, packed as the mode-4 loader packs a local slab:
 *   packed[9*(lx*gridY*gridZ + qz*gridY + qy) + 3*row + col],  lx = q_x - x0.
 * max_entry returns the largest |P_ab(q)| of this slab (a diagnostic of how close I - c_box W comes to singular).
 */
{
	const size_t L[3]={gridX,gridY,gridZ};
	const size_t half[3]={gridX/2,gridY/2,gridZ/2};
	const size_t gYZ=gridY*gridZ;
	const doublecomplex c_box=NPWBoxPolarizability(damping);
	doublecomplex *line,*W,*packed;
	doublecomplex g[6];
	fftw_plan plan_x,plan_yz;
	size_t j[3],x,y,z,lx,yz;
	int comp,n_line,n_plane[2];
	double max_abs=0;

	/* Work arrays: line = 6 components of g along x for one (y,z); W = the 6 components on the local slab,
	 * W[(comp*nx + lx)*gYZ + z*gridY + y]. FFTW plans are made before the arrays are filled.
	 */
	line=(doublecomplex *)voidVector(6*gridX*sizeof(doublecomplex),ALL_POS,"npw generator line");
	W=(doublecomplex *)voidVector(6*nx*gYZ*sizeof(doublecomplex),ALL_POS,"npw symbol slab");
	n_line=(int)gridX;
	plan_x=fftw_plan_many_dft(1,&n_line,6,(fftw_complex *)line,NULL,1,(int)gridX,(fftw_complex *)line,NULL,1,
		(int)gridX,FFTW_FORWARD,FFTW_ESTIMATE);
	n_plane[0]=(int)gridZ; // planes are stored z-major: index z*gridY + y
	n_plane[1]=(int)gridY;
	plan_yz=fftw_plan_many_dft(2,n_plane,(int)(6*nx),(fftw_complex *)W,NULL,1,(int)gYZ,(fftw_complex *)W,NULL,1,
		(int)gYZ,FFTW_FORWARD,FFTW_ESTIMATE);

	// Step 1 and the x part of step 2: generator lines for 0 <= y <= gridY/2, 0 <= z <= gridZ/2, FFT along x
	for (z=0;z<=half[2];z++) for (y=0;y<=half[1];y++) {
		j[1]=y;
		j[2]=z;
		for (x=0;x<=half[0];x++) {
			j[0]=x;
			NPWGenerator(j,L,kd,window,g);
			for (comp=0;comp<6;comp++) line[comp*gridX+x]=g[comp];
		}
		for (x=half[0]+1;x<gridX;x++) for (comp=0;comp<6;comp++) // mirror in x: g(L - x) = sign g(x)
			line[comp*gridX+x]=NPWMirrorSign(comp,0)*line[comp*gridX+(gridX-x)];
		fftw_execute(plan_x);
		/* The x transform commutes with mirroring y and z, so the lines of (gridY - y, z), (y, gridZ - z) and
		 * (gridY - y, gridZ - z) follow from this one by signs. A mirror exists only for 0 < y < gridY/2.
		 */
		{
			const bool mirror_y=(y>0 && y<half[1]);
			const bool mirror_z=(z>0 && z<half[2]);
			int my,mz;
			for (my=0;my<=(mirror_y ? 1 : 0);my++) for (mz=0;mz<=(mirror_z ? 1 : 0);mz++) {
				const size_t yy=my ? gridY-y : y;
				const size_t zz=mz ? gridZ-z : z;
				for (comp=0;comp<6;comp++) {
					const double sign=(my ? NPWMirrorSign(comp,1) : 1.0)*(mz ? NPWMirrorSign(comp,2) : 1.0);
					for (lx=0;lx<nx;lx++) W[(comp*nx+lx)*gYZ+zz*gridY+yy]=sign*line[comp*gridX+x0+lx];
				}
			}
		}
	}
	// The y and z part of step 2: 2D FFT of every (component, q_x) plane
	fftw_execute(plan_yz);
	fftw_destroy_plan(plan_x);
	fftw_destroy_plan(plan_yz);
	Free_cVector(line);

	// Step 3: one 3x3 inverse per frequency, P = (I - c_box W)^-1 (adjugate formula; P is symmetric as W)
	packed=(doublecomplex *)voidVector(9*nx*gYZ*sizeof(doublecomplex),ALL_POS,"npw P slab");
	for (lx=0;lx<nx;lx++) for (yz=0;yz<gYZ;yz++) {
		doublecomplex B[6],adj[6],det;
		doublecomplex *P=packed+9*(lx*gYZ+yz);
		for (comp=0;comp<6;comp++) {
			B[comp]=-c_box*W[(comp*nx+lx)*gYZ+yz];
			if (npw_sym_row[comp]==npw_sym_col[comp]) B[comp]+=1.0;
		}
		// B = [[a b c] [b d e] [c e f]] with a=B0 b=B1 c=B2 d=B3 e=B4 f=B5
		adj[0]=B[3]*B[5]-B[4]*B[4];
		adj[1]=B[2]*B[4]-B[1]*B[5];
		adj[2]=B[1]*B[4]-B[2]*B[3];
		adj[3]=B[0]*B[5]-B[2]*B[2];
		adj[4]=B[1]*B[2]-B[0]*B[4];
		adj[5]=B[0]*B[3]-B[1]*B[1];
		det=B[0]*adj[0]+B[1]*adj[1]+B[2]*adj[2];
		if (det==0) LogError(ALL_POS,"npw preconditioner: I - c_box W(q) is singular (damping s too small?)");
		for (comp=0;comp<6;comp++) {
			const doublecomplex value=adj[comp]/det;
			const int row=npw_sym_row[comp],col=npw_sym_col[comp];
			P[3*row+col]=value;
			P[3*col+row]=value;
			if (cabs(value)>max_abs) max_abs=cabs(value);
		}
	}
	Free_cVector(W);
	*max_entry=max_abs;
	return packed;
}

static void NPWCompareWithFile(const char *filename,const bool distributed)
/* Debug check (ADDA_PRECOND_NPW_COMPARE): the built P(q) against a mode-4 file of the Python exporter. The file is
 * read with the mode-4 loader's own functions; precond is restored afterwards.
 */
{
	const size_t gridN=gridX*gridY*gridZ;
	uint64_t header[5],dims[3];
	double diff2=0,ref2=0,max_diff=0,max_ref=0;
	size_t i,count;
	const doublecomplex *mine,*ref;
	doublecomplex *ref_buf=NULL;
	FILE *f=fopen(filename,"rb");

	if (f==NULL) LogError(ONE_POS,"ADDA_PRECOND_NPW_COMPARE: cannot open '%s'",filename);
	if (fread(header,sizeof(uint64_t),5,f)!=5 || fread(dims,sizeof(uint64_t),3,f)!=3)
		LogError(ONE_POS,"ADDA_PRECOND_NPW_COMPARE: cannot read the header of '%s'",filename);
	if (header[0]!=PRECOND_MAGIC || header[3]!=PRECOND_MODE_FFTDIRECT || dims[0]!=gridX || dims[1]!=gridY
		|| dims[2]!=gridZ)
		LogError(ONE_POS,"ADDA_PRECOND_NPW_COMPARE: '%s' is not a mode-4 file on the grid %zux%zux%zu",filename,
			gridX,gridY,gridZ);
	if (distributed) {
		doublecomplex *built=precond.conv_Phat_local;
		LoadFFTDirectLocalPhat(f,filename,gridN); // sets precond.conv_Phat_local to the file's slab
		ref_buf=precond.conv_Phat_local;
		precond.conv_Phat_local=built;
		mine=built;
		count=9*local_Nx*gridY*gridZ;
	}
	else {
		ref_buf=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"npw compare buffer");
		ReadComplexValues(f,ref_buf,9*gridN,filename,"npw compare Phat");
		mine=precond.conv_Phat;
		count=9*gridN;
	}
	fclose(f);
	ref=ref_buf;
	for (i=0;i<count;i++) {
		const double d=cabs(mine[i]-ref[i]),r=cabs(ref[i]);
		diff2+=d*d;
		ref2+=r*r;
		if (d>max_diff) max_diff=d;
		if (r>max_ref) max_ref=r;
	}
	Free_cVector(ref_buf);
	MyInnerProduct(&diff2,double_type,1,NULL);
	MyInnerProduct(&ref2,double_type,1,NULL);
#ifdef ADDA_MPI
	MPI_Allreduce(MPI_IN_PLACE,&max_diff,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&max_ref,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
	if (IFROOT) printf("NPW COMPARE with '%s': max |diff| %.3e (max |P| %.3e), relative L2 difference %.3e\n",
		filename,max_diff,max_ref,sqrt(diff2/ref2));
}

/* -precond_npw_match (plan task D2m, critical review B1, B2, E2): the same npw kernel, but built from ADDA's own
 * operator instead of the fixed point-dipole + LDR (S = 0) model above:
 *   - G is ADDA's interaction term InterTerm_int (point dipoles, FCD or static FCD, whatever -int selects), so the
 *     kernel matches the matrix ADDA solves also with -int fcd;
 *   - the coupling constant is ADDA's own CoupleConstant for the damped material, for the current polarization
 *     (after every InitCC: the X polarization and any -orient change S in LDR), possibly a diagonal tensor (CLDR).
 * ADDA's scaled matrix is I + cc_sqrt D cc_sqrt with D from InterTerm_int; P(q) = (I - C^1/2 W(q) C^1/2)^-1 with
 * C = diag(c_box) is the same formula as the old build's (I - c_box W)^-1 when c_box is a scalar (the sign and the
 * units are those of the old build, whose generator NPWGreen is InterTerm_core in units of the dipole size).
 * The generator (steps 1-2) does not depend on the polarization, so it is computed once and kept; every update
 * only redoes the 3x3 inverses (step 3). The old -precond_npw path above is not changed.
 */
static bool npw_match_distributed=false; // apply mode chosen by PrecondBuildNPW
static size_t npw_match_x0=0,npw_match_nx=0; // x-slab of this rank (or the whole grid)
static doublecomplex *npw_match_W=NULL; // cached symbol W(q), layout of NPWBuildSlab's W array
static int npw_match_updates=0; // number of builds so far (for the log)

static void NPWGeneratorMatched(const size_t j[3],const size_t L[3],const int window,doublecomplex g[6])
/* Generator g(j) as NPWGenerator, with ADDA's interaction term (integer offsets in units of the dipole size)
 * instead of NPWGreen. The zero offset carries no interaction (InterTerm_int is not defined there).
 */
{
	int t[3][2];
	double h[3][2];
	doublecomplex term[6];
	int axis,ex,ey,ez,comp;

	for (axis=0;axis<3;axis++) {
		t[axis][0]=(int)j[axis];
		t[axis][1]=(int)j[axis]-(int)L[axis];
		h[axis][0]=NPWWindow((double)t[axis][0]/(double)L[axis],window);
		h[axis][1]=NPWWindow((double)t[axis][1]/(double)L[axis],window);
	}
	for (comp=0;comp<6;comp++) g[comp]=0;
	for (ex=0;ex<2;ex++) for (ey=0;ey<2;ey++) for (ez=0;ez<2;ez++) {
		const double weight=h[0][ex]*h[1][ey]*h[2][ez];
		if (weight==0) continue;
		if (t[0][ex]==0 && t[1][ey]==0 && t[2][ez]==0) continue; // G(0) = 0
		(*InterTerm_int)(t[0][ex],t[1][ey],t[2][ez],term);
		for (comp=0;comp<6;comp++) g[comp]+=weight*term[comp];
	}
}

static doublecomplex *NPWSymbolSlabMatched(const size_t x0,const size_t nx,const int window)
/* Steps 1 and 2 of NPWBuildSlab (same loops, mirrors and FFTs) with NPWGeneratorMatched. Returns W with the layout
 * W[(comp*nx + lx)*gridY*gridZ + z*gridY + y].
 */
{
	const size_t L[3]={gridX,gridY,gridZ};
	const size_t half[3]={gridX/2,gridY/2,gridZ/2};
	const size_t gYZ=gridY*gridZ;
	doublecomplex *line,*W;
	doublecomplex g[6];
	fftw_plan plan_x,plan_yz;
	size_t j[3],x,y,z,lx;
	int comp,n_line,n_plane[2];

	line=(doublecomplex *)voidVector(6*gridX*sizeof(doublecomplex),ALL_POS,"npw match generator line");
	W=(doublecomplex *)voidVector(6*nx*gYZ*sizeof(doublecomplex),ALL_POS,"npw match symbol slab");
	n_line=(int)gridX;
	plan_x=fftw_plan_many_dft(1,&n_line,6,(fftw_complex *)line,NULL,1,(int)gridX,(fftw_complex *)line,NULL,1,
		(int)gridX,FFTW_FORWARD,FFTW_ESTIMATE);
	n_plane[0]=(int)gridZ;
	n_plane[1]=(int)gridY;
	plan_yz=fftw_plan_many_dft(2,n_plane,(int)(6*nx),(fftw_complex *)W,NULL,1,(int)gYZ,(fftw_complex *)W,NULL,1,
		(int)gYZ,FFTW_FORWARD,FFTW_ESTIMATE);
	for (z=0;z<=half[2];z++) for (y=0;y<=half[1];y++) {
		j[1]=y;
		j[2]=z;
		for (x=0;x<=half[0];x++) {
			j[0]=x;
			NPWGeneratorMatched(j,L,window,g);
			for (comp=0;comp<6;comp++) line[comp*gridX+x]=g[comp];
		}
		for (x=half[0]+1;x<gridX;x++) for (comp=0;comp<6;comp++)
			line[comp*gridX+x]=NPWMirrorSign(comp,0)*line[comp*gridX+(gridX-x)];
		fftw_execute(plan_x);
		{
			const bool mirror_y=(y>0 && y<half[1]);
			const bool mirror_z=(z>0 && z<half[2]);
			int my,mz;
			for (my=0;my<=(mirror_y ? 1 : 0);my++) for (mz=0;mz<=(mirror_z ? 1 : 0);mz++) {
				const size_t yy=my ? gridY-y : y;
				const size_t zz=mz ? gridZ-z : z;
				for (comp=0;comp<6;comp++) {
					const double sign=(my ? NPWMirrorSign(comp,1) : 1.0)*(mz ? NPWMirrorSign(comp,2) : 1.0);
					for (lx=0;lx<nx;lx++) W[(comp*nx+lx)*gYZ+zz*gridY+yy]=sign*line[comp*gridX+x0+lx];
				}
			}
		}
	}
	fftw_execute(plan_yz);
	fftw_destroy_plan(plan_x);
	fftw_destroy_plan(plan_yz);
	Free_cVector(line);
	return W;
}

static void NPWInvertMatched(const doublecomplex *W,const size_t nx,const doublecomplex c_box[3],doublecomplex *packed,
	double *max_entry)
/* Step 3 with a diagonal coupling tensor C = diag(c_box): P(q) = (I - C^1/2 W(q) C^1/2)^-1, packed as in
 * NPWBuildSlab. For equal components this is (I - c_box W)^-1.
 */
{
	const size_t gYZ=gridY*gridZ;
	doublecomplex sq[3];
	size_t lx,yz;
	int comp;
	double max_abs=0;

	/* Scalar residual around matched NPW. The default is the unmodified inverse. */
	for (comp=0;comp<3;comp++) sq[comp]=csqrt(npw_scale*c_box[comp]);
	for (lx=0;lx<nx;lx++) for (yz=0;yz<gYZ;yz++) {
		doublecomplex B[6],adj[6],det;
		doublecomplex *P=packed+9*(lx*gYZ+yz);
		for (comp=0;comp<6;comp++) {
			B[comp]=-sq[npw_sym_row[comp]]*sq[npw_sym_col[comp]]*W[(comp*nx+lx)*gYZ+yz];
			if (npw_sym_row[comp]==npw_sym_col[comp]) B[comp]+=1.0;
		}
		adj[0]=B[3]*B[5]-B[4]*B[4];
		adj[1]=B[2]*B[4]-B[1]*B[5];
		adj[2]=B[1]*B[4]-B[2]*B[3];
		adj[3]=B[0]*B[5]-B[2]*B[2];
		adj[4]=B[1]*B[2]-B[0]*B[4];
		adj[5]=B[0]*B[3]-B[1]*B[1];
		det=B[0]*adj[0]+B[1]*adj[1]+B[2]*adj[2];
		if (det==0) LogError(ALL_POS,"npw preconditioner (matched): I - C W(q) is singular (damping s too small?)");
		for (comp=0;comp<6;comp++) {
			const doublecomplex value=adj[comp]/det+(npw_sym_row[comp]==npw_sym_col[comp] ? npw_shift : 0.0);
			const int row=npw_sym_row[comp],col=npw_sym_col[comp];
			P[3*row+col]=value;
			P[3*col+row]=value;
			if (cabs(value)>max_abs) max_abs=cabs(value);
		}
	}
	*max_entry=max_abs;
}

static void NPWMatchedSetup(const bool distributed)
/* PrecondBuildNPW for -precond_npw_match: checks, the precond fields, work vectors; the kernel itself is built by
 * PrecondUpdateNPWMatched after each InitCC (ADDA's interaction term is not initialized yet at this point).
 */
{
	if (IntRelation!=G_POINT_DIP && IntRelation!=G_FCD && IntRelation!=G_FCD_ST)
		LogError(ONE_POS,"-precond_npw_match supports only '-int poi', '-int fcd' and '-int fcd_st' (IGT and nloc "
			"interaction terms are not translation-invariant on the integer grid or are not tested)");
	if (surface) LogError(ONE_POS,"-precond_npw_match does not model the reflected interaction (-surf)");
#if !defined(PARALLEL) || defined(SPARSE)
	if (distributed) LogError(ONE_POS,"-precond_npw: distributed apply is available only in the MPI (non-sparse) build");
#endif
	precond.mode=PRECOND_MODE_FFTDIRECT;
	precond.nnz=0;
	precond.conv_gx=gridX;
	precond.conv_gy=gridY;
	precond.conv_gz=gridZ;
	precond.conv_gridN=gridX*gridY*gridZ;
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
	precond.conv_Phat_local=NULL;
	precond.conv_work_in=NULL;
	precond.conv_work_out=NULL;
	precond.conv_plan_fwd=NULL;
	precond.conv_plan_bwd=NULL;
	npw_match_distributed=distributed;
#if defined(PARALLEL) && !defined(SPARSE)
	npw_match_x0=distributed ? local_x0 : 0;
	npw_match_nx=distributed ? local_Nx : gridX;
#else
	npw_match_x0=0;
	npw_match_nx=gridX;
#endif
	if (IFROOT) printf("NPW preconditioner matched to ADDA's operator (-precond_npw_match): window %s, s=%g, grid "
		"%zux%zux%zu, interaction term and coupling constant of ADDA, rebuilt for every polarization, %s\n",
		precond_npw_window==PRECOND_NPW_WINDOW_HANN ? "hann" : "tri",precond_npw_damping,gridX,gridY,gridZ,
		distributed ? "distributed MPI apply" : "replicated apply");
	// The rest as at the end of PrecondBuildNPW
	precond.n=3*nvoid_Ndip;
	precond.tmp=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp");
	precond.tmp2=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp2");
	precond.r_actual=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond r_actual");
	precond.gather_buf=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond gather_buf");
	if (!precond_init_only && IterMethod!=IT_FGMRES && IterMethod!=IT_BICGSTAB && IterMethod!=IT_BICG_CS)
		LogError(ONE_POS,"-precond_npw is applied only by -iter fgmres, -iter bicgstab and -iter bicg (or once, with "
			"-precond_init_only)");
	use_precond=true;
}

#endif // FFTW3

void PrecondUpdateNPWMatched(const doublecomplex c_box[3],const char *label)
/* Build (first call) or rebuild P(q) of -precond_npw_match for the coupling tensor diag(c_box) of the damped material
 * (computed by the caller with ADDA's CoupleConstant for the current polarization). Called after each InitCC.
 */
{
#ifndef FFTW3
	LogError(ONE_POS,"-precond_npw_match requires FFTW3 (%s %g)",label,creal(c_box[0]));
#else
	const size_t gYZ=gridY*gridZ,gridN=gridX*gridY*gridZ;
	double max_entry=0,seconds;
	bool first=(npw_match_W==NULL);
	TIME_TYPE tstart;

	Synchronize();
	tstart=GET_TIME();
	if (first) npw_match_W=NPWSymbolSlabMatched(npw_match_x0,npw_match_nx,precond_npw_window);
	if (npw_match_distributed) {
		if (precond.conv_Phat_local==NULL) precond.conv_Phat_local=(doublecomplex *)voidVector(
			9*npw_match_nx*gYZ*sizeof(doublecomplex),ALL_POS,"npw match Phat slab");
		NPWInvertMatched(npw_match_W,npw_match_nx,c_box,precond.conv_Phat_local,&max_entry);
	}
	else {
		doublecomplex *packed=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"npw match packed");
		size_t x,yz,comp;
		NPWInvertMatched(npw_match_W,gridX,c_box,packed,&max_entry);
		if (precond.conv_Phat==NULL) {
			precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"npw match Phat");
			precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"npw work_in");
			precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"npw work_out");
			precond.conv_plan_fwd=(void *)fftw_plan_dft_3d((int)gridZ,(int)gridY,(int)gridX,
				(fftw_complex *)precond.conv_work_in,(fftw_complex *)precond.conv_work_in,FFTW_FORWARD,FFTW_MEASURE);
			precond.conv_plan_bwd=(void *)fftw_plan_dft_3d((int)gridZ,(int)gridY,(int)gridX,
				(fftw_complex *)precond.conv_work_out,(fftw_complex *)precond.conv_work_out,FFTW_BACKWARD,FFTW_MEASURE);
		}
		for (x=0;x<gridX;x++) for (yz=0;yz<gYZ;yz++) for (comp=0;comp<9;comp++)
			precond.conv_Phat[comp*gridN+yz*gridX+x]=packed[9*(x*gYZ+yz)+comp];
		Free_cVector(packed);
	}
#ifdef ADDA_MPI
	MPI_Allreduce(MPI_IN_PLACE,&max_entry,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
	Synchronize();
#ifdef ADDA_MPI
	seconds=GET_TIME()-tstart;
#else
	seconds=(double)(GET_TIME()-tstart)/CLOCKS_PER_SEC;
#endif
	npw_match_updates++;
	if (IFROOT) printf("NPW matched kernel %s (build %d): c_box="CFORM3V", max |P(q)| %.4e, %.3f s%s\n",label,
		npw_match_updates,REIM3V(c_box),max_entry,seconds,first ? " (incl. the generator)" : "");
#endif
}

#ifdef FFTW3
#include "npw_policy.h"
#endif

void PrecondBuildNPW(void)
/* Build the npw kernel inside ADDA (option -precond_npw) and switch the preconditioner on, as PrecondLoad does for a
 * mode-4 file. Called once after MakeParticle (the FFT grid and the MPI slabs are known then).
 */
{
#ifndef FFTW3
	LogError(ONE_POS,"-precond_npw requires FFTW3");
#else
	const size_t gridN=gridX*gridY*gridZ;
	const bool distributed=ShouldUseConvSAIDistributed();
	const char *compare=getenv("ADDA_PRECOND_NPW_COMPARE");
	double max_entry=0,seconds;
	TIME_TYPE tstart;

	if (precond_filename!=NULL) LogError(ONE_POS,"-precond and -precond_npw cannot be used together");
	if (Nmat!=1 || anisotropy)
		LogError(ONE_POS,"-precond_npw is built for one isotropic material (here %d materials%s)",Nmat,
			anisotropy ? ", anisotropic" : "");
	if (rectDip) LogError(ONE_POS,"-precond_npw does not support rectangular dipoles (-rect_dip)");
	if (gridX<2*(size_t)boxX || gridY<2*(size_t)boxY || gridZ<2*(size_t)boxZ || gridX%2 || gridY%2 || gridZ%2)
		LogError(ONE_POS,"-precond_npw needs an even FFT grid of at least twice the box (grid %zux%zux%zu, box "
			"%ix%ix%i)",gridX,gridY,gridZ,boxX,boxY,boxZ);
	// Task D2m: with -precond_npw_match the kernel is built later from ADDA's own operator (after each InitCC)
	if (precond_npw_match) {
		const char *scale_env=getenv("ADDA_NPW_BOX_SCALE"), *shift_env=getenv("ADDA_NPW_IDENTITY_SHIFT");
		const char *window_env=getenv("ADDA_NPW_WINDOW_SHAPE");
		npw_scale=scale_env ? atof(scale_env) : 1.0;
		npw_shift=shift_env ? atof(shift_env) : 0.0;
		npw_window_shape=window_env ? atof(window_env) : 0.0;
		if (!(npw_scale>=0.5 && npw_scale<=1.5 && npw_shift>=-0.5 && npw_shift<=1.0
				&& npw_window_shape>=-1.0 && npw_window_shape<=3.0)) LogError(ONE_POS,"invalid NPW correction");
		if (getenv("ADDA_NPW_HEX_AUTO")) {
			SelectLearnedNPW(2);
			if (!precond_npw) return;
		}
		else if (getenv("ADDA_NPW_RESIDUAL_AUTO")) {
			if (precond_npw_window!=PRECOND_NPW_WINDOW_HANN || precond_npw_damping!=0.05 || precond_npw_learned)
				LogError(ONE_POS,"NPW correction expects the fixed matched Hann kernel with s=0.05");
			SelectLearnedNPW(true);
		}
		else if (precond_npw_learned) SelectLearnedNPW(false);
		NPWMatchedSetup(distributed);
		return;
	}

	Synchronize();
	tstart=GET_TIME();
	// The same fields LoadFFTDirect sets for a mode-4 file
	precond.mode=PRECOND_MODE_FFTDIRECT;
	precond.nnz=0;
	precond.conv_gx=gridX;
	precond.conv_gy=gridY;
	precond.conv_gz=gridZ;
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
	precond.conv_Phat_local=NULL;
	precond.conv_work_in=NULL;
	precond.conv_work_out=NULL;
	precond.conv_plan_fwd=NULL;
	precond.conv_plan_bwd=NULL;

	if (distributed) {
#if defined(PARALLEL) && !defined(SPARSE)
		precond.conv_Phat_local=NPWBuildSlab(local_x0,local_Nx,precond_npw_damping,precond_npw_window,&max_entry);
#else
		LogError(ONE_POS,"-precond_npw: distributed apply is available only in the MPI (non-sparse) build");
#endif
	}
	else {
		// Sequential (or replicated MPI) apply: the whole grid, stored per component in z, y, x order
		doublecomplex *packed=NPWBuildSlab(0,gridX,precond_npw_damping,precond_npw_window,&max_entry);
		size_t x,yz,comp;
		precond.conv_Phat=(doublecomplex *)voidVector(9*gridN*sizeof(doublecomplex),ALL_POS,"npw Phat");
		for (x=0;x<gridX;x++) for (yz=0;yz<gridY*gridZ;yz++) for (comp=0;comp<9;comp++)
			precond.conv_Phat[comp*gridN+yz*gridX+x]=packed[9*(x*gridY*gridZ+yz)+comp];
		Free_cVector(packed);
		// Work buffers and FFTW plans, as LoadFFTDirect makes them for the replicated FFT apply
		precond.conv_work_in=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"npw work_in");
		precond.conv_work_out=(doublecomplex *)voidVector(3*gridN*sizeof(doublecomplex),ALL_POS,"npw work_out");
		precond.conv_plan_fwd=(void *)fftw_plan_dft_3d((int)gridZ,(int)gridY,(int)gridX,
			(fftw_complex *)precond.conv_work_in,(fftw_complex *)precond.conv_work_in,FFTW_FORWARD,FFTW_MEASURE);
		precond.conv_plan_bwd=(void *)fftw_plan_dft_3d((int)gridZ,(int)gridY,(int)gridX,
			(fftw_complex *)precond.conv_work_out,(fftw_complex *)precond.conv_work_out,FFTW_BACKWARD,FFTW_MEASURE);
	}
#ifdef ADDA_MPI
	MPI_Allreduce(MPI_IN_PLACE,&max_entry,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
	Synchronize();
#ifdef ADDA_MPI
	seconds=GET_TIME()-tstart;
#else
	seconds=(double)(GET_TIME()-tstart)/CLOCKS_PER_SEC;
#endif
	if (IFROOT) {
		const doublecomplex c_box=NPWBoxPolarizability(precond_npw_damping);
		printf("NPW preconditioner built inside ADDA: window %s, s=%g, grid %zux%zux%zu, kd=%.12g, m=%.6g%+.6gi, "
			"c_box=%.8g%+.8gi, max |P(q)| %.4e, %s, %.3f s\n",
			precond_npw_window==PRECOND_NPW_WINDOW_HANN ? "hann" : "tri",precond_npw_damping,gridX,gridY,gridZ,kd,
			creal(ref_index[0]),cimag(ref_index[0]),creal(c_box),cimag(c_box),max_entry,
			distributed ? "distributed MPI apply" : "replicated apply",seconds);
	}
	if (compare!=NULL && compare[0]!='\0') NPWCompareWithFile(compare,distributed);

	// The rest as at the end of PrecondLoad: vector length, work vectors, solver check
	precond.n=3*nvoid_Ndip;
	precond.tmp=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp");
	precond.tmp2=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond tmp2");
	precond.r_actual=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond r_actual");
	precond.gather_buf=(doublecomplex *)voidVector(precond.n*sizeof(doublecomplex),ALL_POS,"precond gather_buf");
	if (!precond_init_only && IterMethod!=IT_FGMRES && IterMethod!=IT_BICGSTAB && IterMethod!=IT_BICG_CS)
		LogError(ONE_POS,"-precond_npw is applied only by -iter fgmres, -iter bicgstab and -iter bicg (or once, with "
			"-precond_init_only)");
	use_precond=true;
#endif
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
	TIME_TYPE tstart,elapsed;

	tstart=GET_TIME();
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
	} else if (IsConvFFTMode()) {
		ApplyConvSAI(in,out,n);
	} else {
		// ILU: solve L*z = in, then U*out = z
		doublecomplex *z=(doublecomplex *)voidVector(n*sizeof(doublecomplex),ALL_POS,"precond ILU z");
		SolveLowerTriangular(precond.L_row_ptr,precond.L_col_idx,precond.L_values,in,z,n);
		SolveUpperTriangular(precond.U_row_ptr,precond.U_col_idx,precond.U_values,z,out,n);
		free(z);
	}
	elapsed=GET_TIME()-tstart;
	Timing_Precond+=elapsed;
	Timing_OneIterPrecond+=elapsed;
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
		} else if (IsConvFFTMode()) {
#ifdef FFTW3
			if (precond.conv_plan_fwd!=NULL)
				fftw_destroy_plan((fftw_plan)precond.conv_plan_fwd);
			if (precond.conv_plan_bwd!=NULL)
				fftw_destroy_plan((fftw_plan)precond.conv_plan_bwd);
			// -precond_npw_match (task D2m): the cached symbol belongs to this grid; a later setup must rebuild it
			free(npw_match_W);
			npw_match_W=NULL;
			npw_match_updates=0;
#endif
			free(precond.conv_stencil);
			free(precond.conv_kernel);
			free(precond.conv_grid_to_dipole);
			free(precond.conv_direct_row_ptr);
			free(precond.conv_direct_dipole);
			free(precond.conv_direct_stencil);
			free(precond.conv_Phat);
			free(precond.conv_Phat_local);
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
