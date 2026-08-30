#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void complex_mul_kernel(
    const float* input, const float* weight, float* output,
    int64_t batch, int64_t input_channels, int64_t output_channels,
    int64_t modes1, int64_t modes2) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * output_channels * modes1 * modes2;
    if (linear >= total) return;
    int64_t remainder = linear;
    const int64_t x = remainder % modes2;
    remainder /= modes2;
    const int64_t y = remainder % modes1;
    remainder /= modes1;
    const int64_t output_channel = remainder % output_channels;
    const int64_t batch_index = remainder / output_channels;

    float real = 0.0f;
    float imag = 0.0f;
    for (int64_t input_channel = 0; input_channel < input_channels; ++input_channel) {
        const int64_t input_index = ((batch_index * input_channels + input_channel) * modes1 + y) * modes2 + x;
        const int64_t weight_index = ((input_channel * output_channels + output_channel) * modes1 + y) * modes2 + x;
        const float input_real = input[2 * input_index];
        const float input_imag = input[2 * input_index + 1];
        const float weight_real = weight[2 * weight_index];
        const float weight_imag = weight[2 * weight_index + 1];
        real += input_real * weight_real - input_imag * weight_imag;
        imag += input_real * weight_imag + input_imag * weight_real;
    }
    const int64_t output_index = ((batch_index * output_channels + output_channel) * modes1 + y) * modes2 + x;
    output[2 * output_index] = real;
    output[2 * output_index + 1] = imag;
}

}  // namespace

torch::Tensor complex_mul_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "complex_mul_forward requires CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kComplexFloat && weight.scalar_type() == at::kComplexFloat,
                "complex_mul_forward currently supports complex64 only");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "expected rank-4 tensors");
    TORCH_CHECK(input.size(1) == weight.size(0) && input.size(2) == weight.size(2) && input.size(3) == weight.size(3),
                "incompatible contraction dimensions");
    input = input.contiguous();
    weight = weight.contiguous();
    auto output = torch::empty({input.size(0), weight.size(1), input.size(2), input.size(3)}, input.options());
    const int64_t total = output.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    complex_mul_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input.data_ptr<c10::complex<float>>()),
        reinterpret_cast<const float*>(weight.data_ptr<c10::complex<float>>()),
        reinterpret_cast<float*>(output.data_ptr<c10::complex<float>>()),
        input.size(0), input.size(1), weight.size(1), input.size(2), input.size(3));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "complex_mul_kernel launch failed");
    return output;
}

