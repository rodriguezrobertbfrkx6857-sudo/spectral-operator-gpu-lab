from __future__ import annotations

import torch


def make_smooth_fields(
    batch: int,
    height: int,
    width: int,
    channels: int = 1,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate deterministic smooth fields without downloading a dataset."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.linspace(0.0, 1.0, height, device=device)
    x = torch.linspace(0.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    fields = []
    for batch_index in range(batch):
        field = torch.zeros((height, width), device=device)
        for component in range(3):
            frequency = torch.randint(1, 5, (), generator=generator).item()
            phase = torch.rand((), generator=generator).item() * 6.283185307179586
            amplitude = 0.25 + torch.rand((), generator=generator).item() * 0.5
            field = field + amplitude * torch.sin(frequency * 6.283185307179586 * xx + phase)
            field = field + (amplitude / 2.0) * torch.cos((frequency + 1) * 6.283185307179586 * yy)
        center_x = torch.rand((), generator=generator).item()
        center_y = torch.rand((), generator=generator).item()
        gaussian = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / 0.04)
        field = field + gaussian
        fields.append(field)
    base = torch.stack(fields).unsqueeze(1)
    return base.repeat(1, channels, 1, 1)

