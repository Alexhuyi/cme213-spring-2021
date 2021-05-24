#include <math_constants.h>

#include "BC.h"
const int numYPerStep = 8;
const int SIDE = 32;
/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    // TODO
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int borderSize = order/2;
    if(tid< nx*ny){
        int tidx = tid%nx + borderSize;
        int tidy = tid/nx + borderSize;
        int index = tidx + gx*tidy;
        next[index]=Stencil<order>(curr+index, gx, xcfl, ycfl);
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int order = params.order();

    int numthreads = 512;
    int numblocks = (nx*ny+numthreads-1)/numthreads;
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch (order) {
            case 2:
                gpuStencilGlobal<2><<<numblocks, numthreads>>>(next_grid.dGrid_, curr_grid.dGrid_,gx, nx, ny, xcfl, ycfl);
                break;
            case 4:
                gpuStencilGlobal<4><<<numblocks, numthreads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            case 8:
                gpuStencilGlobal<8><<<numblocks, numthreads>>>(next_grid.dGrid_, curr_grid.dGrid_,gx, nx, ny, xcfl, ycfl);
                break;
        }
        Grid::swap(curr_grid, next_grid);
        // if (i==0)
        // {
        //     curr_grid.fromGPU();
        //     curr_grid.saveStateToFile("0000.csv");
        // }
        // if (i==1000)
        // {
        //     curr_grid.fromGPU();
        //     curr_grid.saveStateToFile("1000.csv");
        // }
        // if (i==2000)
        // {
        //     curr_grid.fromGPU();
        //     curr_grid.saveStateToFile("2000.csv");
        // }
    }

    check_launch("gpuStencilGlobal");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    int borderSize = order/2;
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x<nx){
        for(int y = tid_y*numYPerStep; y<(tid_y+1)*numYPerStep; y++){
            if( y < ny ){
            int index = tid_x + borderSize + (y+borderSize)*gx;
            next[index]=Stencil<order>(curr+index, gx, xcfl, ycfl);
            }
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);


    // TODO: Declare variables/Compute parameters.
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int order = params.order();
    int thread_num = 512;
    int thread_x = 64;
    int thread_y = (thread_num + thread_x -1)/thread_x;
    dim3 threads(thread_x, thread_y);
    int block_x = (nx+threads.x-1)/threads.x;
    int block_y = (ny+threads.y*numYPerStep-1)/(threads.y*numYPerStep);
    dim3 blocks(block_x, block_y);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch (order) {
            case 2:
                gpuStencilBlock<2,numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,gx, nx, ny, xcfl, ycfl);
                break;
            case 4:
                gpuStencilBlock<4,numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            case 8:
                gpuStencilBlock<8,numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,gx, nx, ny, xcfl, ycfl);
                break;
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
    const int warp_id  = threadIdx.y;
    const int lane     = threadIdx.x;
    __shared__ float block[side*side];
    int borderSize = order/2;
    int tid_x = blockIdx.x * (side-order) + threadIdx.x;
    int tid_y = blockIdx.y * (side-order) + threadIdx.y;
    int b_index = lane + warp_id*side;
    int g_index = tid_x + tid_y*gx;
    for (int i = 0; i < side/ order; ++i){
        if ((tid_x< gx) && ((tid_y+i*order) < gy)) {
            block[b_index+i*order*side] = curr[g_index+i*order*gx];
        }
    }
    __syncthreads();

    for(int i = 0; i < side/ order; ++i) {
        if(( tid_x< gx-borderSize) && (lane >= borderSize) && (lane < side -borderSize) && ( (tid_y + i*order)< gy-borderSize) && ((warp_id+i*order) < side - borderSize) && ( (warp_id+i*order) >= borderSize) ) {
            next[g_index+i*order*gx] = Stencil<order>(block+b_index+i*order*side, side, xcfl, ycfl);;
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    //TODO: Declare variables/Compute parameters.
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int gy = params.gy();
    //block side*side 
    int thread_x = SIDE;
    int thread_y = params.order();
    dim3 threads(thread_x, thread_y);
    int blocks_x = (gx+thread_x-order-1)/(thread_x-order);
    int blocks_y = (gy+SIDE-order-1)/(SIDE-order);
    dim3 blocks(blocks_x, blocks_y);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch (order) {
            case 2:
                gpuStencilShared<SIDE,2><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,gx, gy, xcfl, ycfl);
                break;
            case 4:
                gpuStencilShared<SIDE,4><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
                break;
            case 8:
                gpuStencilShared<SIDE,8><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,gx, gy, xcfl, ycfl);
                break;
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}

