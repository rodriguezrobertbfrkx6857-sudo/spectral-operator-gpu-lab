#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void frequency_mul_kernel(
    const float* input, const float* weight_positive, const float* weight_negative,
    float* output,
    int64_t batch, int64_t input_channels, int64_t output_channels,
    int64_t height, int64_t width_freq, int64_t modes1, int64_t modes2) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * output_channels * height * width_freq;
    if (linear >= total) return;
    int64_t remainder = linear;
    const int64_t x = remainder % width_freq;
    remainder /= width_freq;
    const int64_t y = remainder % height;
    remainder /= height;
    const int64_t output_channel = remainder % output_channels;
    const int64_t batch_index = remainder / output_channels;
    const bool selected = x < modes2 && (y < modes1 || y >= height - modes1);
    if (!selected) return;
    const int64_t weight_y = y < modes1 ? y : y - (height - modes1);
    const float* weight = y < modes1 ? weight_positive : weight_negative;
    float real = 0.0f;
    float imag = 0.0f;
    for (int64_t input_channel = 0; input_channel < input_channels; ++input_channel) {
        const int64_t input_index = ((batch_index * input_channels + input_channel) * height + y) * width_freq + x;
        const int64_t weight_index = ((input_channel * output_channels + output_channel) * modes1 + weight_y) * modes2 + x;
        const float input_real = input[2 * input_index];
        const float input_imag = input[2 * input_index + 1];
        const float weight_real = weight[2 * weight_index];
        const float weight_imag = weight[2 * weight_index + 1];
        real += input_real * weight_real - input_imag * weight_imag;
        imag += input_real * weight_imag + input_imag * weight_real;
    }
    const int64_t output_index = ((batch_index * output_channels + output_channel) * height + y) * width_freq + x;
    output[2 * output_index] = real;
    output[2 * output_index + 1] = imag;
}

}  // namespace

torch::Tensor frequency_mul_forward(torch::Tensor input, torch::Tensor weight_positive,
                                    torch::Tensor weight_negative,
                                    int64_t modes1, int64_t modes2) {
    TORCH_CHECK(input.is_cuda() && weight_positive.is_cuda() && weight_negative.is_cuda(),
                "frequency_mul_forward requires CUDA tensors");
    TORCH_CHECK(input.device() == weight_positive.device() && input.device() == weight_negative.device(),
                "input and weights must be on the same CUDA device");
    TORCH_CHECK(input.scalar_type() == at::kComplexFloat &&
                    weight_positive.scalar_type() == at::kComplexFloat &&
                    weight_negative.scalar_type() == at::kComplexFloat,
                "frequency_mul_forward currently supports complex64 only");
    TORCH_CHECK(input.dim() == 4 && weight_positive.dim() == 4 && weight_negative.dim() == 4,
                "expected rank-4 tensors");
    TORCH_CHECK(input.size(1) == weight_positive.size(0) &&
                    input.size(1) == weight_negative.size(0) &&
                    weight_positive.size(1) == weight_negative.size(1) &&
                    weight_positive.size(2) == modes1 && weight_positive.size(3) == modes2 &&
                    weight_negative.size(2) == modes1 && weight_negative.size(3) == modes2,
                "incompatible frequency dimensions");
    TORCH_CHECK(modes1 > 0 && modes1 <= input.size(2) / 2 && modes2 > 0 &&
                    modes2 <= input.size(3),
                "frequency modes do not fit the input spectrum");
    const c10::cuda::CUDAGuard device_guard(input.device());
    input = input.contiguous();
    weight_positive = weight_positive.contiguous();
    weight_negative = weight_negative.contiguous();
    auto output = torch::zeros(
        {input.size(0), weight_positive.size(1), input.size(2), input.size(3)}, input.options());
    const int64_t total = output.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    frequency_mul_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input.data_ptr<c10::complex<float>>()),
        reinterpret_cast<const float*>(weight_positive.data_ptr<c10::complex<float>>()),
        reinterpret_cast<const float*>(weight_negative.data_ptr<c10::complex<float>>()),
        reinterpret_cast<float*>(output.data_ptr<c10::complex<float>>()),
        input.size(0), input.size(1), weight_positive.size(1), input.size(2), input.size(3), modes1,
        modes2);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "frequency_mul_kernel launch failed");
    return output;
}
