#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include <torch/extension.h>

using namespace fastertransformer;

torch::Tensor rms_norm(const torch::Tensor& input, const torch::Tensor& weight, const double eps)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int          n_dim  = input.dim();
    int          M      = 1;
    auto         out    = torch::zeros_like(input);
    for (int i = 0; i < n_dim - 1; ++i)
        M *= input.size(i);
    int N = input.size(n_dim - 1);
    if (input.dtype() == torch::kFloat16) {
        invokeGeneralT5LayerNorm<__half>((__half*)out.data_ptr(),
                                         (const __half*)input.data_ptr(),
                                         (const __half*)weight.data_ptr(),
                                         nullptr,
                                         eps,
                                         M,
                                         N,
                                         stream);
    }
    else {
        invokeGeneralT5LayerNorm<float>((float*)out.data_ptr(),
                                        (const float*)input.data_ptr(),
                                        (const float*)weight.data_ptr(),
                                        nullptr,
                                        eps,
                                        M,
                                        N,
                                        stream);
    }
    return out;
}

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
        invokeAddBiasResidualLayerNorm((__half*)hidden_states.data_ptr(),
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
        invokeAddBiasResidualLayerNorm((float*)hidden_states.data_ptr(),
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

torch::Tensor silu(const torch::Tensor& input)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int          n_dim  = input.dim();
    int          M      = 1;
    for (int i = 0; i < n_dim - 1; ++i)
        M *= input.size(i);
    int N       = input.size(n_dim - 1);
    int seq_len = input.size(n_dim - 2);
    if (input.dtype() == torch::kFloat16) {
        invokeGenericActivation<SiluActivation, __half, __half>((__half*)input.data_ptr(),
                                                                nullptr /*bias*/,
                                                                nullptr /*gated_weights*/,
                                                                nullptr /*gated_bias*/,
                                                                nullptr /*ia3_tasks*/,
                                                                nullptr /*ia3_weights*/,
                                                                M,
                                                                N,
                                                                0 /*int8_mode*/,
                                                                nullptr /*activation_in*/,
                                                                nullptr /*activation_out*/,
                                                                // 0 /*padding_offset*/,
                                                                // seq_len /*seq_len*/,
                                                                stream);
    }
    else {
        invokeGenericActivation<SiluActivation, float, float>((float*)input.data_ptr(),
                                                              nullptr /*bias*/,
                                                              nullptr /*gated_weights*/,
                                                              nullptr /*gated_bias*/,
                                                              nullptr /*ia3_tasks*/,
                                                              nullptr /*ia3_weights*/,
                                                              M,
                                                              N,
                                                              0 /*int8_mode*/,
                                                              nullptr /*activation_in*/,
                                                              nullptr /*activation_out*/,
                                                              //   0 /*padding_offset*/,
                                                              //   seq_len /*seq_len*/,
                                                              stream);
    }
    return input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm", &rms_norm, "FT rms norm wrapper");
    m.def("add_bias_residual_layernorm", &add_bias_residual_layernorm, "FT add bias layernorm wrapper");
    m.def("silu", &silu, "FT silu wrapper");
}

TORCH_LIBRARY(ft, m)
{
    m.def("rms_norm", rms_norm);
    m.def("add_bias_residual_layernorm", add_bias_residual_layernorm);
    m.def("silu", silu);
}