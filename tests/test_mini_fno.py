import torch

from spectral_gpu.models.mini_fno import MiniFNO
from spectral_gpu.models.synthetic_data import make_smooth_fields


def test_mini_fno_forward_and_backward():
    torch.manual_seed(17)
    model = MiniFNO(width=8, modes1=4, modes2=4, optimized=False, seed=19)
    x = make_smooth_fields(2, 32, 32, seed=23).requires_grad_(True)
    output = model(x)
    assert output.shape == (2, 1, 32, 32)
    loss = output.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_optimized_model_matches_reference_model():
    torch.manual_seed(29)
    reference = MiniFNO(width=8, modes1=4, modes2=4, optimized=False, seed=31)
    optimized = MiniFNO(width=8, modes1=4, modes2=4, optimized=True, seed=31)
    optimized.load_state_dict(reference.state_dict())
    x = make_smooth_fields(1, 32, 32, seed=37)
    torch.testing.assert_close(optimized(x), reference(x), rtol=2.0e-5, atol=2.0e-5)

