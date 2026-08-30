#include <torch/extension.h>

torch::Tensor complex_mul_forward(torch::Tensor input, torch::Tensor weight);
torch::Tensor frequency_mul_forward(torch::Tensor input, torch::Tensor weight,
                                    int64_t modes1, int64_t modes2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("complex_mul_forward", &complex_mul_forward, "Complex channel contraction (CUDA)");
    module.def("frequency_mul_forward", &frequency_mul_forward, "Frequency selection and contraction (CUDA)");
}

