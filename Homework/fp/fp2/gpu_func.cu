#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#define BLOCK_SIZE 16
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 4
/*Algorithm 1*/
__global__
void device_GEMM_1(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K,bool A_T, bool B_T){
    //compute c[i,j]
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if((A_T==false) && (B_T==false)){
      if((row < M) && (col < N)){
        real Cvalue = 0.0;
        for(int k=0; k<K; k++){
          Cvalue += A[row+k*M]*B[k+col*K];
        }
        C[row+col*M] = alpha * Cvalue + beta * C[row+col*M];
      }
    }
    else if((A_T == true) && (B_T == false)){
      if((row < M) && (col < N)){
        real Cvalue = 0.0;
        // real elem_1 = 0.0;
        // real elem_2 = 0.0;
        for(int k=0; k<K; k++){
          // elem_1 = A_T ? A[k + row * K] : A[row+k*M];
          // elem_2 = B_T ? B[col + k * N] : B[k+col*K];
          Cvalue += A[k + row * K]*B[k+col*K];
        }
        C[row+col*M] = alpha * Cvalue + beta * C[row+col*M];
      }
    }
    else if((A_T == false) && (B_T == true)){
      if((row < M) && (col < N)){
        real Cvalue = 0.0;
        for(int k=0; k<K; k++){
          Cvalue += A[row+k*M]*B[col + k * N];
        }
        C[row+col*M] = alpha * Cvalue + beta * C[row+col*M];
      }
    }
}

int myGEMM1(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K, bool A_T, bool B_T) {
  // TODO
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_GEMM_1<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K,A_T,B_T);
  return 0;
}

//transpose the A matrix
__global__
void device_GEMM_1_T_1(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K){
    //compute c[i,j]
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if((row < M) && (col < N)){
      real Cvalue = 0.0;
      // real elem_1 = 0.0;
      // real elem_2 = 0.0;
      for(int k=0; k<K; k++){
        // elem_1 = A_T ? A[k + row * K] : A[row+k*M];
        // elem_2 = B_T ? B[col + k * N] : B[k+col*K];
        Cvalue += A[k + row * K]*B[k+col*K];
      }
      C[row+col*M] = alpha * Cvalue + beta * C[row+col*M];
    }
}

int myGEMM_1T1(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real* alpha, real* beta,
  int M, int N, int K) {
// TODO
int numthreads_x = BLOCK_SIZE;
int numthreads_y = BLOCK_SIZE;
int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
dim3 threads(numthreads_x, numthreads_y);
dim3 blocks(numblocks_x, numblocks_y);
device_GEMM_1_T_1<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K);
return 0;
}

//transpose the B matrix
__global__
void device_GEMM_1_T_2(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K){
    //compute c[i,j]
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if((row < M) && (col < N)){
      real Cvalue = 0.0;
      // real elem_1 = 0.0;
      // real elem_2 = 0.0;
      for(int k=0; k<K; k++){
        // elem_1 = A_T ? A[k + row * K] : A[row+k*M];
        // elem_2 = B_T ? B[col + k * N] : B[k+col*K];
        Cvalue += A[row+k*M]*B[col + k * N];
      }
      C[row+col*M] = alpha * Cvalue + beta * C[row+col*M];
    }
}

int myGEMM_1T2(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real* alpha, real* beta,
  int M, int N, int K) {
// TODO
int numthreads_x = BLOCK_SIZE;
int numthreads_y = BLOCK_SIZE;
int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
dim3 threads(numthreads_x, numthreads_y);
dim3 blocks(numblocks_x, numblocks_y);
device_GEMM_1_T_2<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K);
return 0;
}
/* Helper functions for neural networks */
/*Algorithm 2, shared memory*/
__global__
void device_GEMM_shared(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K, bool A_T, bool B_T){
    //compute c[row,col]
    int g_col = blockIdx.x*blockDim.x + threadIdx.x;
    int g_row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    int row = threadIdx.y;
    real Cvalue = 0;
    // Shared memory used to store Asub and Bsub respectively
    __shared__ real As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ real Bs[BLOCK_SIZE][BLOCK_SIZE+1];
    if( (A_T == false) && (B_T == false)){
      for(int m=0; m < (K+BLOCK_SIZE-1)/BLOCK_SIZE; m++){
        int A_col = m*BLOCK_SIZE + col;
        //load Asub
        if((g_row < M) && (A_col < K)){
          As[row][col] = A[g_row + M*A_col];
        }
        //load Bsub
        int B_row = m*BLOCK_SIZE + row;
        if((B_row < K) && (g_col < N)){
          Bs[row][col] = B[B_row + K*g_col];
        }
        __syncthreads();
        //compute
        if((g_row < M) && (g_col < N)){
          int nums = 0;
          //deal with the last block
          if(K-m*BLOCK_SIZE > BLOCK_SIZE)
            nums = BLOCK_SIZE;
          else
            nums = K - m * BLOCK_SIZE;
          
          for(int e=0; e<nums; e++){
            Cvalue += As[row][e]*Bs[e][col];
          }
        }
        __syncthreads();
      }
    }
    else if((A_T == true) && (B_T == false)){
      for(int m=0; m < (K+BLOCK_SIZE-1)/BLOCK_SIZE; m++){
        int A_col = m*BLOCK_SIZE + col;
        //load Asub(transpose) (g_row,A_col) ->(A_col,g_row)
        if((g_row < M) && (A_col < K)){
          As[row][col] = A[g_row * K + A_col];
        }
        //load Bsub
        int B_row = m*BLOCK_SIZE + row;
        if((B_row < K) && (g_col < N)){
          Bs[row][col] = B[B_row + K*g_col];
        }
        __syncthreads();
        //compute
        if((g_row < M) && (g_col < N)){
          int nums = 0;
          //deal with the last block
          if(K-m*BLOCK_SIZE > BLOCK_SIZE)
            nums = BLOCK_SIZE;
          else
            nums = K - m * BLOCK_SIZE;
          
          for(int e=0; e<nums; e++){
            Cvalue += As[row][e]*Bs[e][col];
          }
        }
        __syncthreads();
      }
    }
    else if((A_T == false) && (B_T == true)){
      for(int m=0; m < (K+BLOCK_SIZE-1)/BLOCK_SIZE; m++){
        int A_col = m*BLOCK_SIZE + col;
        //load Asub
        if((g_row < M) && (A_col < K)){
          As[row][col] = A[g_row + M*A_col];
        }
        //load Bsub(transpose) (B_row,g_col) ->(g_col,B_row)
        int B_row = m*BLOCK_SIZE + row;
        if((B_row < K) && (g_col < N)){
          Bs[row][col] = B[B_row*N + g_col];
        }
        __syncthreads();
  
        //compute
        if((g_row < M) && (g_col < N)){
          int nums = 0;
          //deal with the last block
          if(K-m*BLOCK_SIZE > BLOCK_SIZE)
            nums = BLOCK_SIZE;
          else
            nums = K - m * BLOCK_SIZE;
          
          for(int e=0; e<nums; e++){
            Cvalue += As[row][e]*Bs[e][col];
          }
        }
        __syncthreads();
      }
    };

    //write Csub; Each thread writes one element
    if((g_row < M) && (g_col < N)){
      int index = g_row + M*g_col;
      C[index] = alpha * Cvalue + beta*C[index];
    }
}

int myGEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta, int M, int N, int K, bool A_T, bool B_T) {
  // TODO
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_GEMM_shared<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K,A_T,B_T);
  return 0;
}

__global__
void device_GEMM_shared_T1(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K){
    //compute c[row,col]
    int g_col = blockIdx.x*blockDim.x + threadIdx.x;
    int g_row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    int row = threadIdx.y;
    real Cvalue = 0;
    // Shared memory used to store Asub and Bsub respectively
    __shared__ real As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ real Bs[BLOCK_SIZE][BLOCK_SIZE+1];

    for(int m=0; m < (K+BLOCK_SIZE-1)/BLOCK_SIZE; m++){
      int A_col = m*BLOCK_SIZE + col;
      //load Asub(transpose) (g_row,A_col) ->(A_col,g_row)
      if((g_row < M) && (A_col < K)){
        As[row][col] = A[g_row * K + A_col];
      }
      //load Bsub
      int B_row = m*BLOCK_SIZE + row;
      if((B_row < K) && (g_col < N)){
        Bs[row][col] = B[B_row + K*g_col];
      }
      __syncthreads();
      //compute
      if((g_row < M) && (g_col < N)){
        int nums = 0;
        //deal with the last block
        if(K-m*BLOCK_SIZE > BLOCK_SIZE)
          nums = BLOCK_SIZE;
        else
          nums = K - m * BLOCK_SIZE;
        
        for(int e=0; e<nums; e++){
          Cvalue += As[row][e]*Bs[e][col];
        }
      }
      __syncthreads();
    }
    //write Csub; Each thread writes one element
    if((g_row < M) && (g_col < N)){
      int index = g_row + M*g_col;
      C[index] = alpha * Cvalue + beta*C[index];
    }
}

int myGEMMT1(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K) {
  // TODO
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_GEMM_shared_T1<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K);
  return 0;
}

__global__
void device_GEMM_shared_T2(real* __restrict__ A, real* __restrict__ B,
  real* __restrict__ C, real alpha, real beta,
  int M, int N, int K){
    //compute c[row,col]
    int g_col = blockIdx.x*blockDim.x + threadIdx.x;
    int g_row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    int row = threadIdx.y;
    real Cvalue = 0;
    // Shared memory used to store Asub and Bsub respectively
    __shared__ real As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ real Bs[BLOCK_SIZE][BLOCK_SIZE+1];

    for(int m=0; m < (K+BLOCK_SIZE-1)/BLOCK_SIZE; m++){
      int A_col = m*BLOCK_SIZE + col;
      //load Asub
      if((g_row < M) && (A_col < K)){
        As[row][col] = A[g_row + M*A_col];
      }
      //load Bsub(transpose) (B_row,g_col) ->(g_col,B_row)
      int B_row = m*BLOCK_SIZE + row;
      if((B_row < K) && (g_col < N)){
        Bs[row][col] = B[B_row*N + g_col];
      }
      __syncthreads();

      //compute
      if((g_row < M) && (g_col < N)){
        int nums = 0;
        //deal with the last block
        if(K-m*BLOCK_SIZE > BLOCK_SIZE)
          nums = BLOCK_SIZE;
        else
          nums = K - m * BLOCK_SIZE;
        
        for(int e=0; e<nums; e++){
          Cvalue += As[row][e]*Bs[e][col];
        }
      }
      __syncthreads();
    }
    //write Csub; Each thread writes one element
    if((g_row < M) && (g_col < N)){
      int index = g_row + M*g_col;
      C[index] = alpha * Cvalue + beta*C[index];
    }
}

int myGEMMT2(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K) {
  // TODO
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_GEMM_shared_T2<<<blocks,threads>>>(A,B,C,*alpha,*beta,M,N,K);
  return 0;
}

//a even better GEMM implementation (16,4)
__global__
void device_GEMM_shared2(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta, int M, int N, int K) {
    int g_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.x;
    int row = threadIdx.y;
    int t_row = row * BLOCK_SIZE_X + col; //// map each thread to a int between 0 and BLOCK_SIZE_Y*BLOCK_SIZE_X
    int g_row = blockIdx.y * BLOCK_SIZE_Y * BLOCK_SIZE_X + t_row;

    int Csub_ncols;
    if(N - blockIdx.x * BLOCK_SIZE_X > BLOCK_SIZE_X)
      Csub_ncols = BLOCK_SIZE_X;
    else
      Csub_ncols = N - blockIdx.x * BLOCK_SIZE;
      
    __shared__ real Bs[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    real As[BLOCK_SIZE_Y];
    real Cvalue[BLOCK_SIZE_X]={0};

    for (int m = 0; m < (K+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y; m++)   {
        int Bsub_nrows;
        if( (K - BLOCK_SIZE_Y*m) > BLOCK_SIZE_Y)
          Bsub_nrows = BLOCK_SIZE_Y;
        else
          Bsub_nrows = K - BLOCK_SIZE_Y*m;
        if (((BLOCK_SIZE_Y * m + row) < K) && (g_col < N) ) {
            Bs[row][col]=B[BLOCK_SIZE_Y * m + row + K*g_col];
        }
        
        __syncthreads();
        
        if (g_row<M) {
            for (int i=0; i < BLOCK_SIZE_Y;i++) {
                if ((BLOCK_SIZE_Y*m + i)>=K) {
                    break;
                }
                As[i]=A[g_row+M*(BLOCK_SIZE_Y*m+i)];
            }
        }
        if ((g_row<M)) {           
            for (int i = 0; i < Csub_ncols; i++){
              for(int k = 0; i < Bsub_nrows; k++){
                Cvalue[i]+=As[k]*Bs[k][i];
              }
            }
        }      
        __syncthreads();
    }
    
    if ((row<M)) {
        for (int i=0; i < Csub_ncols; i++) {
            int index =  g_row + M*(blockIdx.x * blockDim.x+i);
            C[index] = alpha*Cvalue[i] + beta*C[index];
        }
    }
}

int myGEMMs2(real* __restrict__ A, real* __restrict__ B, real* __restrict__ C, real* alpha, real* beta, 
  int M, int N, int K) {
    int numthreads_x = BLOCK_SIZE_X;
    int numthreads_y = BLOCK_SIZE_Y;
    int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
    int numblocks_y = (M + numthreads_y*numthreads_x - 1)/(numthreads_y*numthreads_x);
    dim3 threads(numthreads_x, numthreads_y);
    dim3 blocks(numblocks_x, numblocks_y);
    device_GEMM_shared2<<<blocks, threads>>>(A, B, C, *alpha,*beta, M, N, K);
    return 0;
}

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
        mat[row + col * M] = 1 / (real)(1+std::exp(-mat[row + col * M]));
    }
    return;
}

void gpu_sigmoid(real *mat, int M, int N)  {
    int numthreads_x = BLOCK_SIZE;
    int numthreads_y = BLOCK_SIZE;
    int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
    int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
    dim3 threads(numthreads_x, numthreads_y);
    dim3 blocks(numblocks_x, numblocks_y);

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
    dim3 threads(BLOCK_SIZE);

    int numBlocks_x = (N + threads.x -1)/ threads.x;
    dim3 blocks(numBlocks_x);

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
void device_addmat_inplace(real *A, real *B, real alpha, real beta, int M, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < M) && (col < N)) {
      A[row+col*M] = alpha * A[row+col*M]+ beta * B[row+col*M];
  }
}
void gpu_addmat_inplace(real *A, real *B, real alpha, real beta, int M, int N){
  int numthreads_x = BLOCK_SIZE;
  int numthreads_y = BLOCK_SIZE;
  int numblocks_x = (N + numthreads_x - 1)/numthreads_x;
  int numblocks_y = (M + numthreads_y - 1)/numthreads_y;
  dim3 threads(numthreads_x, numthreads_y);
  dim3 blocks(numblocks_x, numblocks_y);
  device_addmat_inplace<<<blocks, threads>>>(A,B, alpha,beta, M, N);
}

__global__
void device_sigmoid_backprop(real* A, real* B, real* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < M) && (col < N)) {
        C[row + col * M] = (real)((A[row + col * M]) * (B[row + col * M]) * (1.0 - B[row + col * M]));
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
void device_row_sum(real* __restrict__ mat1, real* __restrict__ mat2,
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
    dim3 threads(16);
    dim3 blocks(ceil(M / (float)threads.x));
    device_row_sum<<<blocks, threads>>>(mat1, mat2, M, N);
}


__global__
void device_transpose(real* result, real* data, int M, int N)  {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[j + i * N] = data[i + j * M];
    }
    return; 
}

void fastTranspose(real* result, real* data, int M, int N)  {

    int block_size_x = 32;
    int block_size_y = 32;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_transpose<<<blocks, threads>>>(result, data, M, N);
}
