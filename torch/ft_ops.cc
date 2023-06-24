#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include <torch/extension.h>

torch::Tensor add_bias_residual_layernorm(torch::Tensor        hidden_states,
                                 const torch::Tensor& residual,
                                 const torch::Tensor& bias,
                                 const torch::Tensor& gamma,
                                 const torch::Tensor& beta,
                                 const double         eps)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int          n_dim  = hidden_states.dim();
    int          M      = 1;
    for (int i = 0; i < n_dim - 1; ++i)
        M *= hidden_states.size(i);
    int N = hidden_states.size(n_dim - 1);
    if (hidden_states.dtype() == torch::kFloat16) {
        fastertransformer::invokeAddBiasResidualLayerNorm((__half*)hidden_states.data_ptr(),
                                                          (const __half*)residual.data_ptr(),
                                                          (const __half*)bias.data_ptr(),
                                                          (const __half*)gamma.data_ptr(),
                                                          (const __half*)beta.data_ptr(),
                                                          eps,
                                                          M,
                                                          N,
                                                          stream);
    }
    else {
        fastertransformer::invokeAddBiasResidualLayerNorm((float*)hidden_states.data_ptr(),
                                                          (const float*)residual.data_ptr(),
                                                          (const float*)bias.data_ptr(),
                                                          (const float*)gamma.data_ptr(),
                                                          (const float*)beta.data_ptr(),
                                                          eps,
                                                          M,
                                                          N,
                                                          stream);
    }
    return hidden_states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_bias_residual_layernorm", &add_bias_residual_layernorm, "FT add bias layernorm wrapper");
}

TORCH_LIBRARY(ft, m)
{
    m.def("add_bias_residual_layernorm", add_bias_residual_layernorm);
}