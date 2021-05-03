#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = A pi(t) + (1 / (2N))
 *
 */
__global__
void device_graph_propagate(
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {
    // TODO: fill in the kernel code here
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_num = blockDim.x*gridDim.x;
    for(int l=0; l< num_nodes; l+=thread_num){
        int i = tid + l;
        if(i < num_nodes){
            float sum = 0.f;

            // for all of its edges
            for (uint j = graph_indices[i]; j < graph_indices[i + 1]; j++)
            {
                sum += graph_nodes_in[graph_edges[j]] * inv_edges_per_node[graph_edges[j]];
            }
    
            graph_nodes_out[i] = 0.5f / (float)num_nodes + 0.5f * sum;
        }
    }
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,
    int num_nodes,
    int avg_edges
) {
    // TODO: allocate GPU memory
    uint *d_graph_indices;
    uint *d_graph_edges;
    float *d_node_values_input = nullptr; 
    float *d_node_values_output = nullptr;
    float *d_inv_edges_per_node = nullptr;
    const uint num_indices = num_nodes+1;
    const uint num_edges = num_nodes*avg_edges;

    cudaMalloc(&d_graph_indices,num_indices*sizeof(uint));
    cudaMalloc(&d_graph_edges,num_edges*sizeof(uint));
    cudaMalloc(&d_node_values_input,num_nodes*sizeof(float));
    cudaMalloc(&d_node_values_output,num_nodes*sizeof(float));
    cudaMalloc(&d_inv_edges_per_node,num_nodes * sizeof(float));
    // TODO: check for allocation failure
    if (!d_graph_indices || !d_graph_edges || !d_node_values_input || !d_node_values_output || !d_inv_edges_per_node) {
        std::cerr << "Couldn't allocate memory!" << std::endl;
        return 1;
      }
    // TODO: copy data to the GPU
    cudaMemcpy(d_graph_indices,h_graph_indices,num_indices*sizeof(uint),cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges,h_graph_edges,num_edges*sizeof(uint),cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_values_input,h_node_values_input,num_nodes*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_edges_per_node,h_inv_edges_per_node,num_nodes*sizeof(float),cudaMemcpyHostToDevice);
    // launch kernels
    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;
    
    // TODO: launch your kernels the appropriate number of iterations
    int numBlocks = (num_nodes + block_size - 1) / block_size;

    for(int iter = 0; iter < nr_iterations / 2; iter++) 
    {
        device_graph_propagate<<<numBlocks,block_size>>>(d_graph_indices, d_graph_edges, d_node_values_input, d_node_values_output,
                             d_inv_edges_per_node, num_nodes);
        device_graph_propagate<<<numBlocks,block_size>>>(d_graph_indices, d_graph_edges, d_node_values_output, d_node_values_input,
                             d_inv_edges_per_node, num_nodes);
    }

    // handle the odd case and copy memory to the output location
    if (nr_iterations % 2) 
    {
        device_graph_propagate<<<numBlocks,block_size>>>(d_graph_indices, d_graph_edges, d_node_values_input, d_node_values_output,
            d_inv_edges_per_node, num_nodes);
    }
    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    if(nr_iterations %2)
    {
        cudaMemcpy(h_gpu_node_values_output,d_node_values_output,num_nodes * sizeof(float),cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(h_gpu_node_values_output,d_node_values_input,num_nodes * sizeof(float),cudaMemcpyDeviceToHost);
    }
    // TODO: free the memory you allocated!
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(d_node_values_input);
    cudaFree(d_node_values_output);
    cudaFree(d_inv_edges_per_node);

    return gpu_elapsed_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations)
{
    // TODO
    uint each_node_bytes=(2*edges+1)*sizeof(float)+(edges+2)*sizeof(uint);
    return nodes*each_node_bytes*iterations;
}

#endif
