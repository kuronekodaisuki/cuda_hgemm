// gemm_mma_k64_fp16_bf16.cu
// RTX 4060 Ti (sm_89) 実戦版: cp.async(3-stage) + ldmatrix + mma.m16n8k16 (FP16/BF16→FP32)
// 特徴:
//  - TB: 128x128x64, 3-stage pipeline (cp.async.commit/wait <Stages-2> = 1)
//  - Warp: 64x64、mma を 4x8 回、Kステップは 64 = 4 × 16
//  - B共有メモリに列パディング（+8）でバンク競合を緩和
//  - lane→(m,n) 正規マッピングでCへ書き戻し
//  - 参照: --b_colmajor なら B を col-major とみなしてロード経路を切替
//
// 使い方：
//   nvcc -arch=sm_89 -O3 -std=c++17 gemm_mma_k64_fp16_bf16.cu -o gemm_fp16
//   ./gemm_fp16 M N K [--b_colmajor]
//   BF16は -DWMMA_DTYPE_BF16 を付けてビルド
//
// 注意：CUTLASS非依存の最小実戦コード。プロファイルは Nsight Compute 推奨。

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

#define CUDA_CHECK(x) do{ auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

constexpr int TB_M=128, TB_N=128, TB_K=64;    // 3-stage向けにK=64
constexpr int WARP_M=64, WARP_N=64;
constexpr int MMA_M=16, MMA_N=8, MMA_K=16;
constexpr int WARPS_PER_CTA=8, THREADS_PER_CTA=WARPS_PER_CTA*32;
constexpr int STAGES=3;
constexpr int PAD_N=8; // Bタイルに+8列

#ifdef WMMA_DTYPE_BF16
  #define MMA_DTYPE_A "bf16"
  #define MMA_DTYPE_B "bf16"
  using htype = nv_bfloat16;
#else
  #define MMA_DTYPE_A "f16"
  #define MMA_DTYPE_B "f16"
  using htype = __half;
#endif

// ---- cp.async helpers ----
__device__ inline void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(gmem_ptr));
}
__device__ inline void cp_async_commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ inline void cp_async_wait(int n){ asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

// warp座標（2x4=8 warps）
__device__ inline void warp_coords(int &wm, int &wn){
  int w = threadIdx.x >> 5; wm = w/4; wn = w%4;
}

// lane→(row,col) mapping for mma.m16n8k16（各laneが4要素）
__device__ inline void mma16816_lane_map(int lane, int i /*0..3*/, int &row, int &col){
  int g = lane >> 2;  // 0..7
  int t = lane & 3;   // 0..3
  int i_major = i>>1; // 0/1  (col +0/+4)
  int i_minor = i&1;  // 0/1  (row +0/+1)
  row = (g&1)*8 + t*2 + i_minor;                       // 0..15
  col = (g>>1)*2 + (t&1) + i_major*4;                  // 0..7
}

// A/B row-major 前提（B col-major 指定時はロード式を切替）
extern "C" __global__
void gemm_fp_tc_k64_kernel(const htype* __restrict__ A,
                           const htype* __restrict__ B,
                           float* __restrict__ C,
                           int M,int N,int K,int lda,int ldb,int ldc,
                           int b_colmajor)
{
  int block_m = blockIdx.y * TB_M;
  int block_n = blockIdx.x * TB_N;

  extern __shared__ __align__(16) unsigned char smem[];
  htype* As = reinterpret_cast<htype*>(smem);
  htype* Bs = As + STAGES*TB_M*TB_K;
  auto As_stage = [&](int s){ return As + s*TB_M*TB_K; };
  auto Bs_stage = [&](int s){ return Bs + s*TB_K*(TB_N+PAD_N); };

  int tid = threadIdx.x, lane = tid & 31;
  int wM,wN; warp_coords(wM,wN);
  int warp_m0 = wM*WARP_M;
  int warp_n0 = wN*WARP_N;

  float acc[(WARP_M/MMA_M)][(WARP_N/MMA_N)][4];
  #pragma unroll
  for(int i=0;i<WARP_M/MMA_M;i++)
    #pragma unroll
    for(int j=0;j<WARP_N/MMA_N;j++)
      #pragma unroll
      for(int r=0;r<4;r++) acc[i][j][r]=0.f;

  const htype* gA0 = A + block_m*lda;
  const htype* gB0 = B + (b_colmajor ? block_n*ldb : block_n);

  auto prefetch_stage = [&](int k0, int s){
    // A: [TB_M x TB_K]
    int bytesA = TB_M*TB_K*sizeof(htype);
    int chunksA = (bytesA+15)/16;
    for(int c=tid;c<chunksA;c+=THREADS_PER_CTA){
      int byte_off=c*16;
      int e = byte_off/sizeof(htype);
      int r = e/TB_K, kk = e%TB_K;
      const htype* gp = gA0 + r*lda + (k0+kk);
      void* sp = (void*)((char*)As_stage(s)+byte_off);
      bool in = (block_m+r<M)&&(k0+kk<K);
      if(in) cp_async_cg(sp,gp);
      else   asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sp),"l"(gp),"n"(16));
    }
    // B: [TB_K x (TB_N+PAD_N)]
    int bytesB = TB_K*(TB_N+PAD_N)*sizeof(htype);
    int chunksB = (bytesB+15)/16;
    for(int c=tid;c<chunksB;c+=THREADS_PER_CTA){
      int byte_off=c*16;
      int e = byte_off/sizeof(htype);
      int kk = e/(TB_N+PAD_N);
      int nn = e%(TB_N+PAD_N);
      void* sp = (void*)((char*)Bs_stage(s)+byte_off);
      if(nn<TB_N){
        const htype* gp;
        bool in;
        if(!b_colmajor){
          gp = gB0 + (k0+kk)*ldb + nn; // row-major
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

  // stage0
  prefetch_stage(0,0);
  cp_async_wait(STAGES-1); // 3段→wait 2（=全待ち）
  __syncthreads();

  int write_stage=1;
  for(int k0=0;k0<K;k0+=TB_K){
    int read_stage = write_stage^1;
    if(k0+TB_K<K) prefetch_stage(k0+TB_K, write_stage); // 次を先行

    // TB_K=64 → kk=0,16,32,48 の4サブステップ
    #pragma unroll
    for(int kk=0; kk<TB_K; kk+=MMA_K){
      const htype* As_base = As_stage(read_stage) + (warp_m0+0)*TB_K + kk;
      const htype* Bs_base = Bs_stage(read_stage) + kk*(TB_N+PAD_N) + warp_n0;

      #pragma unroll
      for(int i=0;i<WARP_M/MMA_M;i++){
        #pragma unroll
        for(int j=0;j<WARP_N/MMA_N;j++){
          unsigned a0,a1,a2,a3, b0,b1;
          const htype* Ap = As_base + i*MMA_M*TB_K;
          const htype* Bp = Bs_base + j*MMA_N;

          asm volatile(
            "{\n\t"
            ".reg .u64 ra, rb;\n\t"
            "cvta.to.shared.u64 ra, %1;\n\t"
            "cvta.to.shared.u64 rb, %2;\n\t"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%3,%4,%5}, [ra];\n\t"
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%6,%7}, [rb];\n\t"
            "}\n"
            : "=r"(a0), "=l"(Ap), "=l"(Bp), "=r"(a1), "=r"(a2), "=r"(a3), "=r"(b0), "=r"(b1)
          );

          float d0=acc[i][j][0], d1=acc[i][j][1], d2=acc[i][j][2], d3=acc[i][j][3];
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32." MMA_DTYPE_A "." MMA_DTYPE_B ".f32 "
            "{ %0, %1, %2, %3 }, "
            "{ %4, %5, %6, %7 }, "
            "{ %8, %9 }, "
            "{ %0, %1, %2, %3 };\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1)
          );
          acc[i][j][0]=d0; acc[i][j][1]=d1; acc[i][j][2]=d2; acc[i][j][3]=d3;
        }
      }
    }

    if(k0+TB_K<K) cp_async_wait(STAGES-2); // 3段→ wait 1 （= <Stages-2>）
    __syncthreads();
    write_stage ^= 1;
  }

  // Cへ書き戻し（warp毎 64x64）
  int c_row0 = block_m + warp_m0;
  int c_col0 = block_n + warp_n0;
  int lane = threadIdx.x & 31;

  #pragma unroll
  for(int i=0;i<WARP_M/MMA_M;i++){
    #pragma unroll
    for(int j=0;j<WARP_N/MMA_N;j++){
      float d0=acc[i][j][0], d1=acc[i][j][1], d2=acc[i][j][2], d3=acc[i][j][3];
      int r0,c0,r1,c1,r2,c2,r3,c3;
      mma16816_lane_map(lane,0,r0,c0);
      mma16816_lane_map(lane,1,r1,c1);
      mma16816_lane_map(lane,2,r2,c2);
      mma16816_lane_map(lane,3,r3,c3);
      int br = c_row0 + i*MMA_M, bc = c_col0 + j*MMA_N;
      if (br+r0 < M && bc+c0 < N) C[(br+r0)*ldc + (bc+c0)] = d0;
      if (br+r1 < M && bc+c1 < N) C[(br+r1)*ldc + (bc+c1)] = d1;
      if (br+r2 < M && bc+c2 < N) C[(br+r2)*ldc + (bc+c2)] = d2;
      if (br+r3 < M && bc+c3 < N) C[(br+r3)*ldc + (bc+c3)] = d3;
    }
  }
}

// ---- 簡易ドライバ（小さいサイズはCPU検算） ----
static void cpu_ref(const std::vector<htype>& A, const std::vector<htype>& B, std::vector<float>& C,
                    int M,int N,int K,int lda,int ldb,int ldc, bool b_colmajor){
  auto hf = [] __host__ __device__ (htype x){
  #ifdef WMMA_DTYPE_BF16
    return __bfloat162float(x);
  #else
    return __half2float(x);
  #endif
  };
  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      float acc=0.f;
      for(int k=0;k<K;++k){
        float a = hf(A[m*lda+k]);
        float b = b_colmajor? hf(B[n + k*ldb]) : hf(B[k*ldb+n]);
        acc += a*b;
      }
      C[m*ldc+n]=acc;
    }
  }
}

int main(int argc, char** argv){
  int M=2048,N=2048,K=2048; bool b_colmajor=false;
  if(argc>=4){ M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); }
  if(argc>=5 && std::string(argv[4])=="--b_colmajor") b_colmajor=true;

  int lda=K, ldb=b_colmajor?K:N, ldc=N;
  size_t bytesA=size_t(M)*K*sizeof(htype);
  size_t bytesB=size_t(K)*N*sizeof(htype);
  size_t bytesC=size_t(M)*N*sizeof(float);

  std::vector<htype> hA(M*K), hB(K*N);
  std::vector<float> hC(M*N,0.f), hCref;

  std::mt19937 rng(123); std::uniform_real_distribution<float> dist(-1.f,1.f);
  for(int i=0;i<M*K;++i){
  #ifdef WMMA_DTYPE_BF16
    hA[i]= __float2bfloat16(dist(rng));
  #else
    hA[i]= __float2half(dist(rng));
  #endif
  }
  for(int i=0;i<K*N;++i){
  #ifdef WMMA_DTYPE_BF16
    hB[i]= __float2bfloat16(dist(rng));
  #else
    hB[i]= __float2half(dist(rng));
  #endif
  }

  htype *dA,*dB; float* dC;
  CUDA_CHECK(cudaMalloc(&dA,bytesA));
  CUDA_CHECK(cudaMalloc(&dB,bytesB));
  CUDA_CHECK(cudaMalloc(&dC,bytesC));
  CUDA_CHECK(cudaMemcpy(dA,hA.data(),bytesA,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,hB.data(),bytesB,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC,0,bytesC));

  dim3 grid((N+TB_N-1)/TB_N, (M+TB_M-1)/TB_M);
  dim3 block(THREADS_PER_CTA);
  size_t smem_bytes = STAGES*TB_M*TB_K*sizeof(htype) + STAGES*TB_K*(TB_N+PAD_N)*sizeof(htype);

  gemm_fp_tc_k64_kernel<<<grid, block, smem_bytes>>>(dA,dB,dC,M,N,K,lda,ldb,ldc,b_colmajor?1:0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(),dC,bytesC,cudaMemcpyDeviceToHost));

  if(M<=512 && N<=512 && K<=512){
    hCref.assign(M*N,0.f);
    cpu_ref(hA,hB,hCref,M,N,K,lda,ldb,ldc,b_colmajor);
    double max_abs=0, max_rel=0;
    for(int i=0;i<M*N;++i){
      double a=hCref[i], b=hC[i];
      max_abs=std::max(max_abs, std::abs(a-b));
      max_rel=std::max(max_rel, std::abs(a-b)/std::max(1.0, std::abs(a)));
    }
    printf("[FP%s] check: max_abs=%.3e max_rel=%.3e  (M=%d N=%d K=%d, B_col=%d)\n",
      #ifdef WMMA_DTYPE_BF16 ? "BF16":"16"
      #endif
      ,max_abs,max_rel,M,N,K,(int)b_colmajor);
  }else{
    printf("[FP%s] done M=%d N=%d K=%d (B_col=%d). Profile with Nsight Compute.\n",
      #ifdef WMMA_DTYPE_BF16 ? "BF16":"16"
      #endif
      ,M,N,K,(int)b_colmajor);
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}