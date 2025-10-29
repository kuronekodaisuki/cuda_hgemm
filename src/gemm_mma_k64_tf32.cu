// gemm_mma_k64_tf32.cu
// RTX 4060 Ti (sm_89): cp.async 3-stage + mma.m16n8k8 (TF32→FP32)
// TF32は A/B をfloatとして共有メモリへ、mma.m16n8k8 へレジスタから供給。
// 使い方: nvcc -arch=sm_89 -O3 -std=c++17 gemm_mma_k64_tf32.cu -o gemm_tf32

#include <cuda.h>
#include <cstdio>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(x) do{ auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

constexpr int TB_M=128, TB_N=128, TB_K=64;
constexpr int WARP_M=64,  WARP_N=64;
constexpr int MMA_M=16,   MMA_N=8,  MMA_K=8;   // TF32 は k=8
constexpr int WARPS_PER_CTA=8, THREADS_PER_CTA=WARPS_PER_CTA*32;
constexpr int STAGES=3;
constexpr int PAD_N=8;

// cp.async helpers (16B)
__device__ inline void cp_async_cg(void* smem_ptr, const void* gmem_ptr){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr),"l"(gmem_ptr));
}
__device__ inline void cp_async_commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ inline void cp_async_wait(int n){ asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

__device__ inline void warp_coords(int &wm,int &wn){ int w=threadIdx.x>>5; wm=w/4; wn=w%4; }

// lane→(row,col) mapping は m16n8kX 共通
__device__ inline void mma168_lane_map(int lane, int i, int &row, int &col){
  int g=lane>>2, t=lane&3; int i_major=i>>1, i_minor=i&1;
  row = (g&1)*8 + t*2 + i_minor;
  col = (g>>1)*2 + (t&1) + i_major*4;
}

extern "C" __global__
void gemm_tf32_k64_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M,int N,int K,int lda,int ldb,int ldc,
                          int b_colmajor)
{
  int block_m=blockIdx.y*TB_M, block_n=blockIdx.x*TB_N;

  extern __shared__ __align__(16) unsigned char smem[];
  float* As = reinterpret_cast<float*>(smem);
  float* Bs = As + STAGES*TB_M*TB_K;
  auto As_stage=[&](int s){ return As + s*TB_M*TB_K; };
  auto Bs_stage=[&](int s){ return Bs + s*TB_K*(TB_N+PAD_N); };

  int tid=threadIdx.x, lane=tid&31;
  int wM,wN; warp_coords(wM,wN);
  int warp_m0=wM*WARP_M, warp_n0=wN*WARP_N;

  float acc[(WARP_M/MMA_M)][(WARP_N/MMA_N)][4];
  #pragma unroll
  for(int i=0;i<WARP_M/MMA_M;i++)
    #pragma unroll
    for(int j=0;j<WARP_N/MMA_N;j++)
      #pragma unroll
      for(int r=0;r<4;r++) acc[i][j][r]=0.f;

  const float* gA0 = A + block_m*lda;
  const float* gB0 = B + (b_colmajor? block_n*ldb : block_n);

  auto prefetch = [&](int k0,int s){
    // A: [TB_M x TB_K] (float) → 16B チャンク
    int bytesA = TB_M*TB_K*sizeof(float);
    int chunksA = (bytesA+15)/16;
    for(int c=tid;c<chunksA;c+=THREADS_PER_CTA){
      int byte_off=c*16;
      int e=byte_off/sizeof(float);
      int r=e/TB_K, kk=e%TB_K;
      const float* gp = gA0 + r*lda + (k0+kk);
      void* sp=(void*)((char*)As_stage(s)+byte_off);
      bool in=(block_m+r<M)&&(k0+kk<K);
      if(in) cp_async_cg(sp,gp);
      else   asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sp),"l"(gp),"n"(16));
    }
    // B: [TB_K x (TB_N+PAD_N)]
    int bytesB = TB_K*(TB_N+PAD_N)*sizeof(float);
    int chunksB = (bytesB+15)/16;
    for(int c=tid;c<chunksB;c+=THREADS_PER_CTA){
      int byte_off=c*16;
      int e=byte_off/sizeof(float);
      int kk=e/(TB_N+PAD_N);
      int nn=e%(TB_N+PAD_N);
      void* sp=(void*)((char*)Bs_stage(s)+byte_off);
      if(nn<TB_N){
        const float* gp;
        bool in;
        if(!b_colmajor){
          gp = gB0 + (k0+kk)*ldb + nn;          // row-major
          in = (k0+kk<K)&&(block_n+nn<N);
        }else{
          gp = gB0 + (block_n+nn) + (k0+kk)*ldb; // col-major
          in = (k0+kk<K)&&(block_n+nn<N);
        }
        if(in) cp_async_cg(sp,gp);
        else   asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sp),"l"(gp),"n"(16));
      }else{
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sp),"l"(Bs_stage(s)),"n"(16));
      }
    }
    cp_async_commit();
  };

  // stage0 準備
  prefetch(0,0);
  cp_async_wait(STAGES-1);
  __syncthreads();

  int write_stage=1;
  for(int k0=0;k0<K;k0+=TB_K){
    int read_stage=write_stage^1;
    if(k0+TB_K<K) prefetch(k0+TB_K, write_stage);

    // TB_K=64 / MMA_K=8 → 8サブステップ（kk=0..56 step 8）
    #pragma unroll
    for(int kk=0; kk<TB_K; kk+=MMA_K){
      const float* As_base = As_stage(read_stage) + (warp_m0+0)*TB_K + kk;
      const float* Bs_base = Bs_stage(read_stage) + kk*(TB_N+PAD_N) + warp_n0;

      #pragma unroll
      for(int i=0;i<WARP_M/MMA_M;i++){
        #pragma unroll
        for(int j=0;j<WARP_N/MMA_N;j++){
          // A(16x8) と B(8x8) のフラグメントを F32 レジスタで準備
          // 各 lane が持つ要素は固定パターン。ここでは連続4Fロードで簡潔に供給（実運用は手最適化でレジスタ数/衝突を最小化）
          float a[4], b[2]; // 簡略（実際の最適は more registers）
          const float* Ap = As_base + i*MMA_M*TB_K;
          const float* Bp = Bs_base + j*MMA_N;

          // 代表例：lane毎のベクト化ロード（安全のため境界分岐は省略）
          // 実装を分かりやすくするため scalar load にしておく：
          // A: 16x8 → lane毎に4要素（想定の座標に対応する値を拾う）
          // B: 8x8  → lane毎に2要素
          // ※ここは “分かりやすさ優先の正当性版”。最適化余地あり。
          int r0,c0,r1,c1,r2,c2,r3,c3;
          mma168_lane_map(lane,0,r0,c0);
          mma168_lane_map(lane,1,r1,c1);
          mma168_lane_map(lane,2,r2,c2);
          mma168_lane_map(lane,3,r3,c3);
          a[0] = Ap[r0*TB_K + c0];
          a[1] = Ap[r1*TB_K + c1];
          a[2] = Ap[r2*TB_K + c2];
          a[3] = Ap[r3*TB_K + c3];
          // B は 8列×8行で、(k, n)に相当
          int br0,bc0,br1,bc1;
          // 8x8 範囲なので rowは0..7, colは0..7 だが、上の mapping を流用して (r?,c?) のうち rowをk側、colをn側に対応づける
          mma168_lane_map(lane,0,br0,bc0);
          mma168_lane_map(lane,1,br1,bc1);
          b[0] = Bp[br0*(TB_N+PAD_N) + bc0];
          b[1] = Bp[br1*(TB_N+PAD_N) + bc1];

          float d0=acc[i][j][0], d1=acc[i][j][1], d2=acc[i][j][2], d3=acc[i][j][3];
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{ %0, %1, %2, %3 }, "
            "{ %4, %5, %6, %7 }, "
            "{ %8, %9 }, "
            "{ %0, %1, %2, %3 };\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1])
          );
          acc[i][j][0]=d0; acc[i][j][1]=d1; acc[i][j][2]=d2; acc[i][j][3]=d3;
        }
      }
    }

    if(k0+TB_K<K) cp_async_wait(STAGES-2);
    __syncthreads();
    write_stage^=1;
  }

  // C store
  int c_row0 = block_m + warp_m0;
  int c_col0 = block_n + warp_n0;
  #pragma unroll
  for(int i=0;i<WARP_M/MMA_M;i++){
    #pragma unroll
    for(int j=0;j<WARP_N/MMA_N;j++){
      float d0=acc[i][j][0], d1=acc[i][j][1], d2=acc[i][j][2], d3=acc[i][j][3];
      int r0,c0,r1,c1,r2,c2,r3,c3;
      mma168_lane_map(lane,0,r0,c0);
      mma168_lane_map(lane,1,r1,c1);
      mma168_lane_map(lane,2,r2,c2);
      mma168_lane_map(lane,3,r3,c3);
      int br=c_row0 + i*MMA_M, bc=c_col0 + j*MMA_N;
      if (br+r0<M && bc+c0<N) C[(br+r0)*ldc + (bc+c0)] = d0;
      if (br+r1<M && bc+c1<N) C[(br+r1)*ldc + (bc+c1)] = d1;
      if (br+r2<M && bc+c2<N) C[(br+r2)*ldc + (bc+c2)] = d2;
      if (br+r3<M && bc+c3<N) C[(br+r3)*ldc + (bc+c3)] = d3;
    }
  }
}

// --- driver ---
static void cpu_ref(const std::vector<float>& A,const std::vector<float>& B,std::vector<float>& C,
                    int M,int N,int K,int lda,int ldb,int ldc,bool b_colmajor){
  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      double acc=0;
      for(int k=0;k<K;++k){
        acc += A[m*lda+k] * (b_colmajor? B[n + k*ldb] : B[k*ldb + n]);
      }
      C[m*ldc+n]=(float)acc;
    }
  }
}

int main(int argc, char** argv){
  int M=2048,N=2048,K=2048; bool b_col=false;
  if(argc>=4){ M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); }
  if(argc>=5 && std::string(argv[4])=="--b_colmajor") b_col=true;

  int lda=K, ldb=b_col?K:N, ldc=N;
  size_t bytesA=size_t(M)*K*sizeof(float);
  size_t bytesB=size_t(K)*N*sizeof(float);
  size_t bytesC=size_t(M)*N*sizeof(float);

  std::vector<float> hA(M*K), hB(K*N), hC(M*N,0.f), hRef;

  std::mt19937 rng(123); std::uniform_real_distribution<float> dist(-1.f,1.f);
  for(auto& x: hA) x=dist(rng);
  for(auto& x: hB) x=dist(rng);

  float *dA,*dB,*dC;
  CUDA_CHECK(cudaMalloc(&dA,bytesA));
  CUDA_CHECK(cudaMalloc(&dB,bytesB));
  CUDA_CHECK(cudaMalloc(&dC,bytesC));
  CUDA_CHECK(cudaMemcpy(dA,hA.data(),bytesA,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,hB.data(),bytesB,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC,0,bytesC));

  dim3 grid((N+TB_N-1)/TB_N, (M+TB_M-1)/TB_M);
  dim3 block(THREADS_PER_CTA);
  size_t smem_bytes = STAGES*TB_M*TB_K*sizeof(float) + STAGES*TB_K*(TB_N+PAD_N)*sizeof(float);

  gemm_tf32_k64_kernel<<<grid, block, smem_bytes>>>(dA,dB,dC,M,N,K,lda,ldb,ldc,b_col?1:0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(),dC,bytesC,cudaMemcpyDeviceToHost));

  if(M<=512 && N<=512 && K<=512){
    hRef.assign(M*N,0.f);
    cpu_ref(hA,hB,hRef,M,N,K,lda,ldb,ldc,b_col);
    double max_abs=0, max_rel=0;
    for(int i=0;i<M*N;++i){
      double a=hRef[i], b=hC[i];
      max_abs=std::max(max_abs, std::abs(a-b));
      max_rel=std::max(max_rel, std::abs(a-b)/std::max(1.0, std::abs(a)));
    }
    printf("[TF32] check: max_abs=%.3e max_rel=%.3e  (M=%d N=%d K=%d, B_col=%d)\n",
           max_abs,max_rel,M,N,K,(int)b_col);
  }else{
    printf("[TF32] done M=%d N=%d K=%d (B_col=%d). Profile with Nsight Compute.\n",M,N,K,(int)b_col);
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
