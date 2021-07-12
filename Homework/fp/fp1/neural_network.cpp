#include "neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "iomanip"
#include "mpi.h"
#include "utils/common.h"

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

real norms(NeuralNetwork& nn) {
  real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
  arma::Mat<real> A, B, C, D;

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
  real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
  real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
  real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
  real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

  int ow = 15;

  if (iter == 0) {
    error_file << std::left << std::setw(ow) << "Iteration" << std::left
               << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow)
               << "Max Err W1" << std::left << std::setw(ow) << "Max Err b0"
               << std::left << std::setw(ow) << "Max Err b1" << std::left
               << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
               << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
               << std::left << std::setw(ow) << "L2 Err b1"
               << "\n";
  }

  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow)
             << max_errW0 << std::left << std::setw(ow) << max_errW1
             << std::left << std::setw(ow) << max_errb0 << std::left
             << std::setw(ow) << max_errb1 << std::left << std::setw(ow)
             << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left
             << std::setw(ow) << L2_errb0 << std::left << std::setw(ow)
             << L2_errb1 << "\n";
}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<real> da1 = nn.W[1].t() * diff;

  arma::Mat<real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg) {
  int N = yc.n_cols;
  real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  real data_loss = ce_sum / N;
  real reg_loss = 0.5 * reg * norms(nn);
  real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads) {
  real h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        real oldval = nn.W[i](j, k);
        nn.W[i](j, k) = oldval + h;
        feedforward(nn, X, numcache);
        real fxph = loss(nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward(nn, X, numcache);
        real fxnh = loss(nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

    for (int j = 0; j < nn.b[i].size(); ++j) {
      real oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward(nn, X, numcache);
      real fxph = loss(nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward(nn, X, numcache);
      real fxnh = loss(nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network nn
 */
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        write_cpudata_tofile(nn, iter);
      }

      iter++;
    }
  }
}

/*
 * TODO
 * Train the neural network nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
struct NNcache{
 real *W1, *W2, *b1, *b2;
 real *a1, *y_pred;
 real *dW1, *dW2, *db1, *db2;
 real *da1, *dz1, *diff;// diff derivative of cross entropy
  /*
  M:number of featrues,
  H:number of neurons in hidden layer,
  C:number of classes, 10
  */
 NNcache(int M, int H, int C, int batch_size){
   cudaMalloc((void**)&W1,sizeof(real)*H*M);
   cudaMalloc((void**)&W2,sizeof(real)*C*H);
   cudaMalloc((void**)&b1,sizeof(real)*H);
   cudaMalloc((void**)&b2,sizeof(real)*C);
   cudaMalloc((void**)&a1,sizeof(real)*H*batch_size);
   cudaMalloc((void**)&y_pred,sizeof(real)*C*batch_size);
   cudaMalloc((void**)&diff,sizeof(real)*C*batch_size);
   cudaMalloc((void**)&dW1,sizeof(real)*H*M);
   cudaMalloc((void**)&dW2,sizeof(real)*C*H);
   cudaMalloc((void**)&db1,sizeof(real)*H);
   cudaMalloc((void**)&db2,sizeof(real)*C);
   cudaMalloc((void**)&da1,sizeof(real)*H*batch_size);
   cudaMalloc((void**)&dz1,sizeof(real)*H*batch_size);
 }

 ~NNcache(){
   cudaFree(W1);
   cudaFree(W2);
   cudaFree(b1);
   cudaFree(b2);
   cudaFree(a1);
   cudaFree(y_pred);
   cudaFree(diff);
   cudaFree(dW1);
   cudaFree(dW2);
   cudaFree(db1);
   cudaFree(db2);
   cudaFree(da1);
   cudaFree(dz1);
 }
};

void parallel_feedforward(NeuralNetwork &nn, real *d_X, NNcache &nncache, int size_per_proc){
    int M = nn.H[0];
    int H = nn.H[1];
    int C = nn.H[2];
    real alpha = 1.0, beta = 1.0;

    /*layer 1  z1 = W1 * X + arma::repmat(b1, 1, N); a1 = sigmoid(z1)*/
    gpu_repmat(nncache.b1, nncache.a1, H, size_per_proc);
    myGEMM(nncache.W1, d_X, nncache.a1, &alpha,&beta, H, size_per_proc, M );
    gpu_sigmoid(nncache.a1,H,size_per_proc);
    /*layer 2 z2 = W2 * a1 + arma::repmat(b2, 1, N); y_pred = a2 = softmax(z2)*/
    gpu_repmat(nncache.b2, nncache.y_pred, C, size_per_proc);
    myGEMM(nncache.W2, nncache.a1, nncache.y_pred, &alpha,&beta, C, size_per_proc, H);
    gpu_softmax(nncache.y_pred, C, size_per_proc);
}

void parallel_backprop(NeuralNetwork& nn, real *d_X, real *d_Y, real reg, NNcache &nncache,int batch_size, int size_per_proc, int num_procs){
    int M = nn.H[0];
    int H = nn.H[1];
    int C = nn.H[2];
    real ratio = 1.0/(real)batch_size;
    reg = reg /num_procs; //change it in the parallel_train functino
    //reg = 0.0;
    /*diff = (1.0 / N) * (bpcache.yc - y)*/
    gpu_addmat(nncache.y_pred,d_Y,nncache.diff, ratio, -ratio, C, size_per_proc);

    /*bpgrads.dW[2] = diff * bpcache.a[1].t() + reg * nn.W[2];*/
    real alpha = 1.0;
    cudaMemcpy(nncache.dW2, nncache.W2, sizeof(real) * C * H, cudaMemcpyDeviceToDevice);
    myGEMMT(nncache.diff,nncache.a1,nncache.dW2,&alpha,&reg,C,H,size_per_proc,false,true);

    /*db2 = arma::sum(diff, 1)*/
    gpu_row_sum(nncache.diff, nncache.db2, C, size_per_proc);

    /* da1 = nncache.W2.t() * diff;*/
    real beta = 0.0;
    myGEMMT(nncache.W2,nncache.diff,nncache.da1,&alpha,&beta,H,size_per_proc,C,true,false);

    /* dz1 = da1 .* nncache.a1 .* (1 - nncache.a1);*/
    gpu_sigmoid_backprop(nncache.da1,nncache.a1,nncache.dz1,H,size_per_proc);

    /*dW1 = dz1 * X.t() + reg * nncache.W1;*/
    cudaMemcpy(nncache.dW1, nncache.W1, sizeof(real) * H * M, cudaMemcpyDeviceToDevice);
    myGEMMT(nncache.dz1, d_X, nncache.dW1, &alpha, &reg, H, M, size_per_proc,false,true);

    /* db1 = arma::sum(dz1, 1);*/
    gpu_row_sum(nncache.dz1, nncache.db1, H, size_per_proc);
}

void parallel_gradientdecent(NeuralNetwork& nn, NNcache &nncache, real learning_rate){
    int M = nn.H[0];
    int H = nn.H[1];
    int C = nn.H[2];

    //update nncache params can change two mat params in-place
    gpu_addmat(nncache.W1,nncache.dW1,nncache.W1,1.0,-learning_rate,H,M);
    gpu_addmat(nncache.W2,nncache.dW2,nncache.W2,1.0,-learning_rate,C,H);
    gpu_addmat(nncache.b1,nncache.db1,nncache.b1,1.0,-learning_rate,H,1);
    gpu_addmat(nncache.b2,nncache.db2,nncache.b2,1.0,-learning_rate,C,1);
}

void parallel_train(NeuralNetwork& nn, const arma::Mat<real>& X,
                    const arma::Mat<real>& y, real learning_rate, real reg,
                    const int epochs, const int batch_size, bool grad_check,
                    int print_every, int debug) {
  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = (rank == 0) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  std::ofstream error_file;
  error_file.open("Outputs/CpuGpuDiff.txt");
  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function. */

  // TODO
  /*
  M:number of featrues,
  H:number of neurons in hidden layer,
  C:number of classes, 10
  */
  int M = nn.H[0];
  int H = nn.H[1];
  int C = nn.H[2];

  int num_batches = (N + batch_size - 1) / batch_size;
  std::vector<real *> d_X_batches(num_batches);
  std::vector<real *> d_Y_batches(num_batches);
  
  //subdivide input batch of images and `MPI_scatter()' to each MPI node
  for (int batch = 0; batch < num_batches; ++batch) {
      int start_col = batch*batch_size;
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      int this_batch_size = last_col - start_col + 1;
      int nsample_per_proc = (this_batch_size + num_procs -1) / num_procs;

      int scounts_X[num_procs], scounts_Y[num_procs], displs_X[num_procs], displs_Y[num_procs];

      for(int i = 0; i < num_procs; i++){
        scounts_X[i] = M*std::min(nsample_per_proc,this_batch_size - i*nsample_per_proc);
        scounts_Y[i] = C*std::min(nsample_per_proc,this_batch_size - i*nsample_per_proc);
        displs_X[i] = i*M*nsample_per_proc;
        displs_Y[i] = i*C*nsample_per_proc;
      }
    
      arma::Mat<real> X_batch(M,scounts_X[rank]/M);
      MPI_SAFE_CALL(MPI_Scatterv(X.colptr(start_col),scounts_X,displs_X,MPI_FP,X_batch.memptr(),
      scounts_X[rank],MPI_FP,0,MPI_COMM_WORLD));
      arma::Mat<real> Y_batch(C ,scounts_Y[rank]/C);
      MPI_SAFE_CALL(MPI_Scatterv(y.colptr(start_col),scounts_Y,displs_Y,MPI_FP,Y_batch.memptr(),
      scounts_Y[rank],MPI_FP,0,MPI_COMM_WORLD));

      // data host to device , 3 processors
      cudaMalloc((void **)&d_X_batches[batch], scounts_X[rank] * sizeof(real));
      cudaMalloc((void **)&d_Y_batches[batch], scounts_Y[rank] * sizeof(real));
      cudaMemcpy(d_X_batches[batch], X_batch.memptr(), scounts_X[rank]* sizeof(real), cudaMemcpyHostToDevice);
      cudaMemcpy(d_Y_batches[batch], Y_batch.memptr(), scounts_Y[rank] * sizeof(real), cudaMemcpyHostToDevice); 
  }
  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;
  //allocate deivice memory for nn parameters and host memory for derivatives
  NNcache nncache(M,H,C,batch_size);
  real *h_dW1 = (real *)malloc(H * M * sizeof(real));
  real *h_dW2 = (real *)malloc(C * H * sizeof(real));
  real *h_db1 = (real *)malloc(H * sizeof(real));
  real *h_db2 = (real *)malloc(C * sizeof(real));

  //copy data from host to devices
  cudaMemcpy(nncache.W1, nn.W[0].memptr(),H*M*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(nncache.W2, nn.W[1].memptr(),C*H*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(nncache.b1, nn.b[0].memptr(),H*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(nncache.b2, nn.b[1].memptr(),C*sizeof(real),cudaMemcpyHostToDevice);

  
  for (int epoch = 0; epoch < epochs; ++epoch) {

    for (int batch = 0; batch < num_batches; ++batch) {
      /*
       * Possible implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */

      // TODO
      int start_col = batch*batch_size;
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      int this_batch_size = last_col - start_col + 1;
      int nsample_per_proc = (this_batch_size + num_procs -1) / num_procs;
      //used for 3 GPU
      int nsample_this_proc = std::min(nsample_per_proc,this_batch_size-rank*nsample_per_proc);
            
      //training
      //forwards
      parallel_feedforward(nn,d_X_batches[batch] ,nncache,nsample_this_proc);

      //backprop
      parallel_backprop(nn,d_X_batches[batch], d_Y_batches[batch],reg, nncache, this_batch_size, nsample_this_proc, num_procs);

      // cudaMemcpy's, cudaMemcpyDeviceToHost
      cudaMemcpy(h_dW1, nncache.dW1, H * M * sizeof(real), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_dW2, nncache.dW2, C * H * sizeof(real), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_db1, nncache.db1, H * sizeof(real), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_db2, nncache.db2, C * sizeof(real), cudaMemcpyDeviceToHost);

      // // MPI_Allreduce
      arma::Mat<real> dW1(size(nn.W[0]), arma::fill::zeros);
      MPI_SAFE_CALL(MPI_Allreduce(h_dW1, dW1.memptr(), H * M, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      arma::Mat<real> dW2(size(nn.W[1]), arma::fill::zeros);
      MPI_SAFE_CALL(MPI_Allreduce(h_dW2, dW2.memptr(), C * H, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      arma::Col<real> db1(size(nn.b[0]), arma::fill::zeros);
      MPI_SAFE_CALL(MPI_Allreduce(h_db1, db1.memptr(), H, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      arma::Col<real> db2(size(nn.b[1]), arma::fill::zeros);
      MPI_SAFE_CALL(MPI_Allreduce(h_db2, db2.memptr(), C, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

      // // cudaMemcpy's, cudaMemcpyHostToDevice
      cudaMemcpy(nncache.dW1, dW1.memptr(), H * M * sizeof(real), cudaMemcpyHostToDevice);
      cudaMemcpy(nncache.dW2, dW2.memptr(), C * H * sizeof(real), cudaMemcpyHostToDevice);
      cudaMemcpy(nncache.db1, db1.memptr(), H * sizeof(real), cudaMemcpyHostToDevice);
      cudaMemcpy(nncache.db2, db2.memptr(), C * sizeof(real), cudaMemcpyHostToDevice);
      //add regularization term
    //  gpu_addmat(nncache.dW1,nncache.W1,nncache.dW1,1.0,reg,H,M);
     // gpu_addmat(nncache.dW2,nncache.W2,nncache.dW2,1.0,reg,C,H);
      // // Gradient descent step
      // nn.W[0] -= learning_rate * dW1;
      // nn.W[1] -= learning_rate * dW2;
      // nn.b[0] -= learning_rate * db1;
      // nn.b[1] -= learning_rate * db2;
      parallel_gradientdecent(nn,nncache,learning_rate);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && rank == 0 && print_flag) {
        // TODO
        // Copy data back to the CPU
        cudaMemcpy(nn.W[0].memptr(), nncache.W1, H * M * sizeof(real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), nncache.W2, C * H * sizeof(real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), nncache.b1, H * sizeof(real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), nncache.b2, C * sizeof(real), cudaMemcpyDeviceToHost);

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */
        write_diff_gpu_cpu(nn, iter, error_file);
      }

      iter++;
    }
  }

  // TODO
  // Copy data back to the CPU
  cudaMemcpy(nn.W[0].memptr(), nncache.W1, H * M * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), nncache.W2, C * H * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[0].memptr(), nncache.b1, H * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), nncache.b2, C * sizeof(real), cudaMemcpyDeviceToHost);
  error_file.close();

  // TODO
  // Free memory
  free(h_dW1);
  free(h_dW2);
  free(h_db1);
  free(h_db2);
  for(int batch = 0; batch < num_batches; ++batch) {
    cudaFree(d_X_batches[batch]);
    cudaFree(d_Y_batches[batch]);
  }
}

// cudaMalloc's
// MPI_Scatter. See this page for details on this function: https://www.open-mpi.org/doc/v4.1/
// cudaMemcpy's, cudaMemcpyHostToDevice
// loop over epochs and batches
// here you use your GPU kernels including myGEMM, sigmoid, softmax, etc
// cudaMemcpy's, cudaMemcpyDeviceToHost
// MPI_Allreduce
// cudaMemcpy's, cudaMemcpyHostToDevice
// Gradient descent step
// At the end, you copy back the nn coefficients from GPU to CPU and run some cudaFree's.

