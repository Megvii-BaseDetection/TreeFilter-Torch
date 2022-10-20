#include <math.h>
#include <thread>
#include <vector>
#include <deque>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_NUM_THREADS         64
#define GET_CUDA_CHANNEL(N)      ceil(512.0f / N)

template <typename scalar_t>
__global__ void root_leaf_prop_kernel(
        scalar_t * in_data, 
        scalar_t * out_data, 
        scalar_t * weight,
        int * sorted_index, 
        int * sorted_parent_index, 
        int batch_size, 
        int channel_size, 
        int vertex_count){

    const int thread_idx    = threadIdx.x;
    const int batch_idx     = blockIdx.x;
    const int channel_idx   = blockIdx.y;
    const int thread_count  = blockDim.x;
    const int channel_step  = gridDim.y;

    in_data             += batch_idx * vertex_count * channel_size;
    out_data            += batch_idx * vertex_count * channel_size;
    weight              += batch_idx * vertex_count;
    sorted_index        += batch_idx * vertex_count;
    sorted_parent_index += batch_idx * vertex_count;

    __shared__ int node_per_thread[CUDA_NUM_THREADS];
    node_per_thread[thread_idx] = -1;
    if (thread_idx == 0){
        weight[0]              = 0;
        sorted_parent_index[0] = 0;
    }
    __syncthreads();

    int i = thread_idx;
    while (i < vertex_count){
        int par = sorted_parent_index[i];
        int par_thread = par % thread_count;
        if ((node_per_thread[par_thread] >= par) || (i == 0)){
            int cur_pos = sorted_index[i];
            int par_pos = sorted_index[par];
            for (int k = channel_idx * vertex_count; k < channel_size * vertex_count;
                       k += channel_step * vertex_count){
                scalar_t edge_weight = weight[i];
                out_data[cur_pos + k] = in_data[i + k] * (1 - edge_weight * edge_weight) +
                                        out_data[par_pos + k] * edge_weight;
                __threadfence_block();
            }
            node_per_thread[thread_idx] = i;
            i += thread_count;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void leaf_root_aggr_kernel(
        scalar_t * in_data, 
        scalar_t * out_data, 
        scalar_t * weight,
        int * sorted_index, 
        int * sorted_child_index, 
        int batch_size, 
        int channel_size, 
        int vertex_count,
        int max_adj_per_node){

    const int thread_idx    = threadIdx.x;
    const int batch_idx     = blockIdx.x;
    const int channel_idx   = blockIdx.y;
    const int thread_count  = blockDim.x;
    const int channel_step  = gridDim.y;
    
    if (in_data != NULL){
        in_data    += batch_idx * vertex_count * channel_size;
    }    
    out_data             += batch_idx * vertex_count * channel_size;
    weight               += batch_idx * vertex_count;
    sorted_index         += batch_idx * vertex_count;
    sorted_child_index   += batch_idx * vertex_count * max_adj_per_node;

    __shared__ int node_per_thread[CUDA_NUM_THREADS];
    node_per_thread[thread_idx] = vertex_count;
    __syncthreads();

    int i = vertex_count - thread_idx - 1;
    while (i >= 0){
        int child_len = 0;
        bool valid = true;
        for (int j = 0; j < max_adj_per_node; j++){
            int child        = sorted_child_index[i * max_adj_per_node + j];
            int child_thread = (vertex_count - child - 1) % thread_count;

            if (child <= 0) break;
            if (node_per_thread[child_thread] > child){
                valid = false;
                break;
            }
            child_len++;
        }
        if (valid){
            int cur_pos = sorted_index[i];
            for (int k = channel_idx * vertex_count; k < channel_size * vertex_count; 
                    k += channel_step * vertex_count){
                scalar_t aggr_sum;
                if (in_data != NULL)    
                    aggr_sum = in_data[cur_pos + k];
                else
                    aggr_sum = 1;
                for (int j = 0; j < child_len; j++){
                    int child = sorted_child_index[i * max_adj_per_node + j];
                    aggr_sum += out_data[child + k] * weight[child];
                }
                out_data[i + k] = aggr_sum;
            }
            node_per_thread[thread_idx] = i;
            i -= thread_count;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void root_leaf_grad_kernel(
        scalar_t * in_data,
        scalar_t * in_grad,
        scalar_t * out_data,
        scalar_t * out_grad, 
        scalar_t * weight,
        scalar_t * grad,
        int * sorted_index, 
        int * sorted_parent_index, 
        int batch_size, 
        int data_channel_size,
        int grad_channel_size,
        int vertex_count){

    const int thread_idx    = threadIdx.x;
    const int batch_idx     = blockIdx.x;
    const int channel_idx   = blockIdx.y;
    const int thread_count  = blockDim.x;
    const int channel_step  = gridDim.y;
    const int channel_size  = data_channel_size > grad_channel_size ? data_channel_size : grad_channel_size;

    in_data             += batch_idx * vertex_count * data_channel_size;
    in_grad             += batch_idx * vertex_count * grad_channel_size;
    out_data            += batch_idx * vertex_count * data_channel_size;
    out_grad            += batch_idx * vertex_count * grad_channel_size;
    weight              += batch_idx * vertex_count;
    grad                += batch_idx * vertex_count * channel_size;
    sorted_index        += batch_idx * vertex_count;
    sorted_parent_index += batch_idx * vertex_count;

    __shared__ int node_per_thread[CUDA_NUM_THREADS];
    node_per_thread[thread_idx] = -1;

    int i = thread_idx;
    while (i < vertex_count){
        int cur         = i;
        int par         = sorted_parent_index[i];
        int par_pos     = sorted_index[par];
        int par_thread  = par % thread_count;
        if ((cur == 0) || (node_per_thread[par_thread] >= par)){
            for (int k = channel_idx; k < channel_size; k += channel_step){
                scalar_t edge_weight   = weight[i];
                int data_offset     = (k % data_channel_size) * vertex_count;
                int grad_offset     = (k % grad_channel_size) * vertex_count;
                int out_offset      = k * vertex_count;
                
                if (cur > 0){
                    scalar_t left  = in_grad[cur + grad_offset] * (out_data[par_pos + data_offset] - edge_weight * in_data[cur + data_offset]);
                    scalar_t right = in_data[cur + data_offset] * (out_grad[par + grad_offset] - edge_weight * in_grad[cur + grad_offset]);

                    grad[cur + out_offset]      = left + right;
                    out_grad[cur + grad_offset] = in_grad[cur + grad_offset] * (1 - edge_weight * edge_weight) +
                                                  out_grad[par + grad_offset] * edge_weight;
                    __threadfence_block();
                }
                else
                    grad[cur + out_offset] = 0;
            }
            node_per_thread[thread_idx] = i;
            i += thread_count;
        }
        __syncthreads();
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
refine_forward(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor 
    ){
    
    const int batch_size        = feature_in_tensor.size(0);
    const int channel_size      = feature_in_tensor.size(1); 
    const int vertex_size       = feature_in_tensor.size(2);
    const int max_adj_per_node  = sorted_child_tensor.size(2);

    auto options                  = feature_in_tensor.options();
    auto feature_aggr_tensor      = at::zeros_like(feature_in_tensor, options);
    auto feature_aggr_up_tensor   = at::zeros_like(feature_in_tensor, options);
    auto weight_sum_tensor        = at::zeros({batch_size, vertex_size}, options);
    auto weight_sum_up_tensor     = at::zeros({batch_size, vertex_size}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feature_in_tensor.scalar_type(), "refine_forward", [&] {
        scalar_t * feature_in          = feature_in_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * edge_weight         = edge_weight_tensor.contiguous().data_ptr<scalar_t>();
        int * sorted_index             = sorted_index_tensor.contiguous().data_ptr<int>();
        int * sorted_parent_index      = sorted_parent_tensor.contiguous().data_ptr<int>();
        int * sorted_child_index       = sorted_child_tensor.contiguous().data_ptr<int>();
        scalar_t * feature_aggr        = feature_aggr_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * feature_aggr_sum    = feature_aggr_up_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * weight_sum          = weight_sum_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * weight_aggr_sum     = weight_sum_up_tensor.contiguous().data_ptr<scalar_t>();

        dim3 feature_block_dims(CUDA_NUM_THREADS, 1, 1), feature_grid_dims(batch_size, channel_size, 1);
        leaf_root_aggr_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                feature_in, feature_aggr_sum, edge_weight, sorted_index, sorted_child_index, batch_size, channel_size, vertex_size, max_adj_per_node);
        root_leaf_prop_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                feature_aggr_sum, feature_aggr, edge_weight, sorted_index, sorted_parent_index, batch_size, channel_size, vertex_size);

        dim3 weight_block_dims(CUDA_NUM_THREADS, 1, 1), weight_grid_dims(batch_size, 1, 1);
        leaf_root_aggr_kernel <<< weight_grid_dims, weight_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                static_cast<scalar_t *>(NULL), weight_aggr_sum, edge_weight, sorted_index, sorted_child_index, batch_size, 1, vertex_size, max_adj_per_node);
        root_leaf_prop_kernel <<< weight_grid_dims, weight_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                weight_aggr_sum, weight_sum, edge_weight, sorted_index, sorted_parent_index, batch_size, 1, vertex_size);
    });

    auto feature_out_tensor = feature_aggr_tensor / weight_sum_tensor.unsqueeze(1); 
    auto result = std::make_tuple(feature_out_tensor, feature_aggr_tensor, feature_aggr_up_tensor,
            weight_sum_tensor, weight_sum_up_tensor);

    return result;
}

at::Tensor refine_backward_feature(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor,
        const at::Tensor & feature_out_tensor,
        const at::Tensor & feature_aggr_tensor,
        const at::Tensor & feature_aggr_up_tensor,
        const at::Tensor & weight_sum_tensor,
        const at::Tensor & weight_sum_up_tensor,
        const at::Tensor & grad_out_tensor
    ){

    auto options                        = feature_in_tensor.options();
    auto grad_feature_tensor            = at::zeros_like(feature_in_tensor, options);
    auto grad_feature_aggr_sum_tensor   = at::zeros_like(feature_in_tensor, options);

    auto grad_out_norm_tensor = grad_out_tensor / weight_sum_tensor.unsqueeze(1);

    const int batch_size        = feature_in_tensor.size(0);
    const int channel_size      = feature_in_tensor.size(1); 
    const int vertex_size       = feature_in_tensor.size(2);
    const int max_adj_per_node  = sorted_child_tensor.size(2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feature_in_tensor.scalar_type(), "refine_backward_feature", [&] {
        scalar_t * feature_in          = feature_in_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * edge_weight         = edge_weight_tensor.contiguous().data_ptr<scalar_t>();
        int * sorted_index             = sorted_index_tensor.contiguous().data_ptr<int>();
        int * sorted_parent_index      = sorted_parent_tensor.contiguous().data_ptr<int>();
        int * sorted_child_index       = sorted_child_tensor.contiguous().data_ptr<int>();
        scalar_t * feature_aggr        = feature_aggr_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * feature_aggr_sum    = feature_aggr_up_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * weight_sum          = weight_sum_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * weight_aggr_sum     = weight_sum_up_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_out            = grad_out_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_feature        = grad_feature_tensor.contiguous().data_ptr<scalar_t>();

        scalar_t * grad_out_norm           = grad_out_norm_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_feature_aggr_sum   = grad_feature_aggr_sum_tensor.contiguous().data_ptr<scalar_t>();

        dim3 feature_block_dims(CUDA_NUM_THREADS, 1, 1), feature_grid_dims(batch_size, channel_size, 1);
        leaf_root_aggr_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                grad_out_norm, grad_feature_aggr_sum, edge_weight, sorted_index, sorted_child_index, batch_size, channel_size, vertex_size, max_adj_per_node);
        root_leaf_prop_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                grad_feature_aggr_sum, grad_feature, edge_weight, sorted_index, sorted_parent_index, batch_size, channel_size, vertex_size);
    });

    return grad_feature_tensor;
}

at::Tensor refine_backward_weight(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor,
        const at::Tensor & feature_out_tensor,
        const at::Tensor & feature_aggr_tensor,
        const at::Tensor & feature_aggr_up_tensor,
        const at::Tensor & weight_sum_tensor,
        const at::Tensor & weight_sum_up_tensor,
        const at::Tensor & grad_out_tensor
    ){

    auto options            = feature_in_tensor.options();
    auto grad_weight_tensor = at::zeros_like(edge_weight_tensor, options);

    const int batch_size        = feature_in_tensor.size(0);
    const int channel_size      = feature_in_tensor.size(1); 
    const int vertex_size       = feature_in_tensor.size(2);
    const int max_adj_per_node  = sorted_child_tensor.size(2);
        
    auto grad_all_channel_tensor        = at::zeros_like(feature_in_tensor, options);
    auto grad_norm_all_channel_tensor   = at::zeros_like(feature_in_tensor, options);
    auto grad_out_norm_aggr_sum_tensor  = at::zeros_like(feature_in_tensor, options);
    auto feature_grad_aggr_sum_tensor   = at::zeros_like(feature_in_tensor, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feature_in_tensor.scalar_type(), "refine_backward_weight", [&] {
        scalar_t * feature_in          = feature_in_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * edge_weight         = edge_weight_tensor.contiguous().data_ptr<scalar_t>();
        int * sorted_index             = sorted_index_tensor.contiguous().data_ptr<int>();
        int * sorted_parent_index      = sorted_parent_tensor.contiguous().data_ptr<int>();
        int * sorted_child_index       = sorted_child_tensor.contiguous().data_ptr<int>();
        scalar_t * feature_out         = feature_out_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * feature_aggr        = feature_aggr_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * feature_aggr_sum    = feature_aggr_up_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * weight_sum          = weight_sum_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * weight_aggr_sum     = weight_sum_up_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_out            = grad_out_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_weight         = grad_weight_tensor.contiguous().data_ptr<scalar_t>();
        
        scalar_t * grad_all_channel            = grad_all_channel_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_norm_all_channel       = grad_norm_all_channel_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_out_norm_aggr_sum      = grad_out_norm_aggr_sum_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * feature_grad_aggr_sum       = feature_grad_aggr_sum_tensor.contiguous().data_ptr<scalar_t>();

        auto grad_out_norm_tensor = grad_out_tensor / weight_sum_tensor.unsqueeze(1);
        auto feature_grad_tensor  = grad_out_norm_tensor * feature_out_tensor; 
        scalar_t * grad_out_norm     = grad_out_norm_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * feature_grad      = feature_grad_tensor.contiguous().data_ptr<scalar_t>();

        dim3 feature_block_dims(CUDA_NUM_THREADS, 1, 1), feature_grid_dims(batch_size, channel_size, 1);
        leaf_root_aggr_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                grad_out_norm, grad_out_norm_aggr_sum, edge_weight, sorted_index, sorted_child_index, batch_size, channel_size, vertex_size, max_adj_per_node);
        leaf_root_aggr_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                feature_grad, feature_grad_aggr_sum, edge_weight, sorted_index, sorted_child_index, batch_size, channel_size, vertex_size, max_adj_per_node);

        root_leaf_grad_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                feature_aggr_sum, grad_out_norm_aggr_sum, feature_aggr, grad_out_norm_aggr_sum, edge_weight, grad_all_channel, 
                sorted_index, sorted_parent_index, batch_size, channel_size, channel_size, vertex_size);
        root_leaf_grad_kernel <<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                weight_aggr_sum, feature_grad_aggr_sum, weight_sum, feature_grad_aggr_sum, edge_weight, grad_norm_all_channel, 
                sorted_index, sorted_parent_index, batch_size, 1, channel_size, vertex_size);

    });

    grad_weight_tensor = (grad_all_channel_tensor - grad_norm_all_channel_tensor).sum(1);

    return grad_weight_tensor;
}
