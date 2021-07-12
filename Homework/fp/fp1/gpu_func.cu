#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#define BLOCK_SIZE 32

__global__
void device_GEMM(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K, bool A_T = false, bool B_T = false){
    //compute c[i,j]
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if((row < M) && (col < N)){
      real Cvalue = 0;
      real elem_1 = 0;
      real elem_2 = 0;
      for(int k=0; k<K; k++){
        elem_1 = A_T ? A[k + row * K] : A[row+k*M];
        elem_2 = B_T ? B[col + k * N] : B[k+col*K];
        Cvalue += elem_1*elem_2;
      }
      C[row+col*M] = alpha * Cvalue + beta * C[row+col*M];
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K) {
  // TODO
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_GEMM<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K);
  return 0;
}

int myGEMMT(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real* alpha, real* beta,
  int M, int N, int K, bool A_T, bool B_T) {
// TODO
int numthreads_x = BLOCK_SIZE;
int numthreads_y = BLOCK_SIZE;
int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
dim3 threads(numthreads_x, numthreads_y);
dim3 blocks(numblocks_x, numblocks_y);
device_GEMM<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K, A_T, B_T);
return 0;
}
/* Helper functions for neural networks */

// TODO
__global__
void device_repmat(real *vec, real *result, int M, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < M) && (col < N)) {
      result[row + col * M] = vec[row];
  }
}

void gpu_repmat(real *vec, real *result, int M, int N){
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_repmat<<<blocks, threads>>>(vec,result, M, N);
}

__global__
void device_sigmoid(real *mat, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < M) && (col < N)) {
        mat[row + col * M] = 1.0 / (1.0+std::exp(-mat[row + col * M]));
    }
    return;
}

void gpu_sigmoid(real *mat, int M, int N)  {

    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_sigmoid<<<blocks, threads>>>(mat, M, N);
}

__global__
void device_softmax(real *mat, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        real sum = 0.0;
        for (int c = 0; c < M; ++c) {
            int index = M * col + c;
            mat[index] = std::exp(mat[index]);
            sum += mat[index];
        }
        for (int c = 0; c < M; ++c) {
            int index = M * col + c;
            mat[index] /= sum;
        }
    }
}

void gpu_softmax(real *mat, int M, int N) {
    dim3 threads(BLOCK_SIZE,1);

    int numBlocks_x = (N + threads.x -1)/ threads.x;
    dim3 blocks(numBlocks_x,1);

    device_softmax<<<blocks, threads>>>(mat, M, N);
};

__global__
void device_addmat(real *A, real *B, real *C, real alpha, real beta, int M, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < M) && (col < N)) {
      C[row+col*M] = alpha * A[row+col*M]+ beta * B[row+col*M];
  }
}
void gpu_addmat(real *A, real *B, real *C, real alpha, real beta, int M, int N){
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_addmat<<<blocks, threads>>>(A,B,C, alpha,beta, M, N);
}

__global__
void device_sigmoid_backprop(real* A, real* B, real* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < M) && (col < N)) {
        C[row + col * M] = (real)((A[row + col * M]) * (B[row + col * M]) * (1.0 - B[row + col * M]))
    }
}


void gpu_sigmoid_backprop(real* A, real* B, real* C, int M, int N) {
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_sigmoid_backprop<<<blocks, threads>>>(A, B, C, M, N);
}

__global__
void device_row_sum(real* mat1, real* mat2,
                        int M, int N) {
    int row =  blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        real sum = 0.0;
        for (size_t col = 0; col < N; ++col) {
            sum += mat1[M * col + row];
        }
        mat2[row] = sum;
    }
}

void gpu_row_sum(real *mat1, real *mat2, int M, int N) {
    int numthreads_x = 1024;
    int numthreads_y = 1;
    int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
    int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_row_sum<<<blocks, threads>>>(mat1, mat2, M, N);
}
