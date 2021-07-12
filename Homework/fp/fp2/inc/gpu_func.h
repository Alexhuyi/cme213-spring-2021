#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "../utils/types.h"

struct event_pair {
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair* p) {
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair* p) {
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}
/*Algorithm 1*/
int myGEMM1(real* __restrict__ A, real* __restrict__ B, real* __restrict__ C, real* alpha, real* beta, int M, int N, int K,bool A_T=false, bool B_T=false);
int myGEMM1T1(real* __restrict__ A, real* __restrict__ B, real* __restrict__ C, real* alpha, real* beta, int M, int N, int K);
int myGEMM1T2(real* __restrict__ A, real* __restrict__ B, real* __restrict__ C, real* alpha, real* beta,int M, int N, int K);

/*Algorithm 2*/
int myGEMM(real* __restrict__ A, real* __restrict__ B,real* __restrict__ C, real* alpha, real* beta, int M, int N, int K,bool A_T=false, bool B_T=false);
int myGEMMT1(real* __restrict__ A, real* __restrict__ B,real* __restrict__ C, real* alpha, real* beta, int M, int N, int K);
int myGEMMT2(real* __restrict__ A, real* __restrict__ B,real* __restrict__ C, real* alpha, real* beta, int M, int N, int K);

/*16*4*/
int myGEMMs2(real* __restrict__ A, real* __restrict__ B, real* __restrict__ C, real* alpha, real* beta, 
  int M, int N, int K);
void gpu_repmat(real *vec, real *result, int M, int N);
void gpu_sigmoid(real *mat, int M, int N);
void gpu_softmax(real *mat, int M, int N);
void gpu_addmat(real *A, real *B, real *C, real alpha, real beta, int M, int N);
void gpu_addmat_inplace(real *A, real *B, real alpha, real beta, int M, int N);
void gpu_sigmoid_backprop(real* A, real* B, real* C, int M, int N);
void gpu_row_sum(real *mat1, real *mat2, int M, int N);
void fastTranspose(real* array_out, real* array_in, int n_rows, int n_cols);
#endif
