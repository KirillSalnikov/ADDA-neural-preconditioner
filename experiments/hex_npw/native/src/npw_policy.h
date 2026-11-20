/* Evaluate the geometry-conditioned NPW policy on ADDA's occupied voxels, before NPW setup. */
#include "npw_weights.h"
#include "npw_residual_weights.h"
#include "hex_weights.h"
#include "hex_time_gate.h"
#include <limits.h>

static int PolicyCoordinate(size_t i,int dim)
{
    int coordinate=(int)position[3*i+dim];
#ifdef PARALLEL
    if (dim==2) coordinate+=local_z0;
#endif
    return coordinate;
}

static void PolicyLinear(const float *input,int nin,float *output,int nout,const float *weights,
    const float *bias,bool activation)
{
    for (int o=0;o<nout;o++) {
        float value=bias[o];
        for (int j=0;j<nin;j++) value+=weights[o*nin+j]*input[j];
        output[o]=activation ? tanhf(value) : value;
    }
}

static double PolicySquaredDifference(float a,float b)
{
    double d=a-b;
    return d*d;
}

static void CalibrateNPW(const float *geo,const float *phys,float *scores)
{
    if (anchor_mix==0) return;
    double distance[NPW_ANCHORS],smallest=HUGE_VAL,mass=0,local[7]={0};
    for (int k=0;k<NPW_ANCHORS;k++) {
        double d=0;
        for (int i=0;i<3;i++) d+=6*PolicySquaredDifference(geo[i],anchor_geo[k][i]);
        d+=10*PolicySquaredDifference(geo[3],anchor_geo[k][3]);
        for (int i=4;i<68;i++) d+=4.0/64*PolicySquaredDifference(geo[i],anchor_geo[k][i]);
        const double scales[]={1,.35,.18,.7};
        for (int i=0;i<4;i++) d+=scales[i]*PolicySquaredDifference(phys[i],anchor_phys[k][i]);
        distance[k]=d;
        if (d<smallest) smallest=d;
    }
    const float baseline=scores[0];
    for (int k=0;k<NPW_ANCHORS;k++) {
        double w=exp(-(distance[k]-smallest)/anchor_bandwidth);
        mass+=w;
        for (int a=0;a<7;a++) local[a]+=w*anchor_cost[k][a];
    }
    for (int a=0;a<7;a++) scores[a]=(1-anchor_mix)*(scores[a]-baseline)+anchor_mix*local[a]/mass;
}

static void SelectLearnedNPW(int correct)
{
    extern double dpl;
    if (correct==2) {
        extern const enum sh shape;
        extern const double sh_pars[];
        const double m=creal(ref_index[0]);
        if (shape!=SH_PRISM || sh_pars[0]!=6. || sh_pars[1]<.65 || sh_pars[1]>1.1 ||
            boxX<18 || boxX>24 || m<1.75 || m>3.25 || fabs(dpl-30.)>1e-8 ||
            IntRelation!=G_FCD || PolRelation!=POL_FCD || cimag(ref_index[0])!=0. ||
            IterMethod!=IT_BICGSTAB || (getenv("ADDA_NPW_TIME_GATE") && nprocs!=1)) {
            if (IFROOT) printf("Hex NPW: outside calibrated range; using requested analytical NPW settings.\n");
            return;
        }
    }
    if (IntRelation!=G_FCD || PolRelation!=POL_FCD || cimag(ref_index[0])!=0)
        LogError(ONE_POS,"The learned NPW policy is trained for real isotropic material and -int fcd -pol fcd");
    int lo[3]={INT_MAX,INT_MAX,INT_MAX},hi[3]={-1,-1,-1},span[3],width=0;
    for (size_t i=0;i<local_nvoid_Ndip;i++) for (int a=0;a<3;a++) {
        int v=PolicyCoordinate(i,a);
        if (v<lo[a]) lo[a]=v;
        if (v>hi[a]) hi[a]=v;
    }
#ifdef PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,lo,3,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,hi,3,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
    for (int a=0;a<3;a++) {
        span[a]=hi[a]-lo[a]+1;
        if (span[a]>width) width=span[a];
    }
    double bins[64]={0};
    for (size_t i=0;i<local_nvoid_Ndip;i++) {
        int bin=0;
        for (int a=0;a<3;a++) bin=4*bin+(4*(PolicyCoordinate(i,a)-lo[a])+2)/width;
        bins[bin]++;
    }
#ifdef PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,bins,64,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif
    float geo[68],input[17],hidden[24],scores[7];
    for (int a=0;a<3;a++) geo[a]=(float)span[a]/boxX;
    geo[3]=(float)(nvoid_Ndip/((double)span[0]*span[1]*span[2]));
    for (int i=0;i<64;i++) geo[i+4]=(float)(64*bins[i]/((double)width*width*width));
    double m=creal(ref_index[0]);
    input[0]=m-2.5;
    input[1]=log2(boxX/16.0);
    input[2]=(dpl-25)/10.0;
    input[3]=log1p(fabs(m-1)*boxX/dpl);
    input[4]=10*m/dpl;
    if (correct==2) {
        if (getenv("ADDA_NPW_TIME_GATE")) {
            const double material=m-2.5, size=((double)boxX-21.)/3., aspect=geo[2]-.85;
            const double f[]={1.,material,size,aspect,material*size,material*aspect};
            double benefit=0.;
            for (int i=0;i<6;i++) benefit+=gate_weight[i]*f[i];
            if (benefit<=0.) {
                precond_npw=precond_npw_match=false;
                if (IFROOT) printf("Hex NPW action -1 (skip kernel construction)\n");
                return;
            }
        }
        const int actions=sizeof(hex_actions)/sizeof(hex_actions[0]);
        float x[67], cost[sizeof(hex_actions)/sizeof(hex_actions[0])]={0};
        for (int i=0;i<64;i++) x[i]=geo[i+4];
        x[64]=geo[2]; x[65]=(boxX-21)/3.0; x[66]=(m-2.5)/.6;
#if HEX_RBF
        const int n=sizeof(centers)/sizeof(centers[0]);
        double distance[sizeof(centers)/sizeof(centers[0])],closest=HUGE_VAL;
        for (int j=0;j<n;j++) {
            double d=0;
            for (int i=0;i<64;i++) d+=PolicySquaredDifference(x[i],centers[j][i])*3./64.;
            d+=4*PolicySquaredDifference(x[64],centers[j][64])+.6*PolicySquaredDifference(x[65],centers[j][65])
                +PolicySquaredDifference(x[66],centers[j][66]);
            distance[j]=d;
            if (d<closest) closest=d;
        }
        for (int j=0;j<n;j++) {
            const double w=exp(-(distance[j]-closest)/HEX_WIDTH);
            for (int a=0;a<actions;a++) cost[a]+=w*costs[j][a];
        }
#else
        float joined[8], hidden_x[16];
        PolicyLinear(x,64,joined,5,&embedding_weight[0][0],embedding_bias,true);
        for (int i=0;i<3;i++) joined[5+i]=x[64+i];
        PolicyLinear(joined,8,hidden_x,16,&hidden_weight[0][0],hidden_bias,true);
        PolicyLinear(hidden_x,16,cost,actions,&cost_weight[0][0],cost_bias,false);
#endif
        int best=0;
        for (int a=1;a<actions;a++) if (cost[a]<cost[best]) best=a;
        precond_npw_window=hex_actions[best][0] ? PRECOND_NPW_WINDOW_HANN : PRECOND_NPW_WINDOW_TRI;
        precond_npw_damping=hex_actions[best][1];
        npw_scale=hex_actions[best][2]; npw_window_shape=hex_actions[best][3];
        if (IFROOT) printf("Hex NPW action %d\n",best);
        return;
    }
    if (correct) {
        const int actions=sizeof(corr_actions)/sizeof(corr_actions[0]);
        float joined[13], hidden_corr[16], cost[sizeof(corr_actions)/sizeof(corr_actions[0])];
        for (int i=0;i<5;i++) joined[i]=input[i];
        PolicyLinear(geo,68,joined+5,8,&corr_encoder_weight[0][0],corr_encoder_bias,true);
        PolicyLinear(joined,13,hidden_corr,16,&corr_hidden_weight[0][0],corr_hidden_bias,true);
        PolicyLinear(hidden_corr,16,cost,actions,&corr_score_weight[0][0],corr_score_bias,false);
        int best=0;
        for (int i=1;i<actions;i++) if (cost[i]<cost[best]) best=i;
        npw_scale=corr_actions[best][0];
        npw_shift=corr_actions[best][1];
        npw_window_shape=corr_actions[best][2];
        if (IFROOT) printf("NPW correction action %d (scale %g, shift %g, window %g)\n",
                           best,npw_scale,npw_shift,npw_window_shape);
        return;
    }
    PolicyLinear(geo,68,input+5,12,&encoder_0_weight[0][0],encoder_0_bias,true);
    PolicyLinear(input,17,hidden,24,&head_0_weight[0][0],head_0_bias,true);
    PolicyLinear(hidden,24,scores,7,&head_2_weight[0][0],head_2_bias,false);
    CalibrateNPW(geo,input,scores);
    int action=0;
    for (int i=1;i<7;i++) if (scores[i]<scores[action]) action=i;
    precond_npw_damping=learned_damping[action];
    precond_npw_window=learned_hann[action] ? PRECOND_NPW_WINDOW_HANN : PRECOND_NPW_WINDOW_TRI;
    if (IFROOT) printf("Learned NPW action %d, damping %.4g, window %s\n",action,
        precond_npw_damping,learned_hann[action] ? "hann" : "tri");
}
