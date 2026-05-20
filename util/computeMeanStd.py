import os
import torch


def computeMeanStd(
    dataloader,
    cuda=False,
    stats_path="normalization_stats.pt",
    force_recompute=False
):
    """
    Compute or load per-channel mean and std.

    Args:
        dataloader: DataLoader returning images with shape (B, C, H, W).
        cuda: If True, use CUDA when available.
        stats_path: Path where normalization statistics are saved/loaded.
        force_recompute: If True, recompute even if stats_path exists.

    Returns:
        mean, std: CPU tensors with shape (C,).
    """

    if os.path.exists(stats_path) and not force_recompute:
        stats = torch.load(stats_path, map_location="cpu")

        if "mean" not in stats or "std" not in stats:
            raise ValueError(
                f"Stats file {stats_path} does not contain 'mean' and 'std'."
            )

        mean = stats["mean"]
        std = stats["std"]

        return mean, std

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    channel_sum = None
    channel_sum_sq = None
    num_pixels = 0

    for data, _ in dataloader:
        data = data.to(device)

        # expected shape: (B, C, H, W)
        if data.dim() != 4:
            raise ValueError(
                f"Expected data with shape (B, C, H, W), got {tuple(data.shape)}"
            )

        b, c, h, w = data.shape

        if channel_sum is None:
            channel_sum = torch.zeros(c, device=device, dtype=torch.float64)
            channel_sum_sq = torch.zeros(c, device=device, dtype=torch.float64)

        data = data.to(torch.float64)

        # sum over batch and spatial dims -> per-channel
        channel_sum += data.sum(dim=(0, 2, 3))
        channel_sum_sq += (data ** 2).sum(dim=(0, 2, 3))
        num_pixels += b * h * w

    if num_pixels == 0:
        raise ValueError("No data found in dataloader")

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)

    mean = mean.cpu()
    std = std.cpu()

    os.makedirs(os.path.dirname(stats_path), exist_ok=True) if os.path.dirname(stats_path) else None

    torch.save(
        {
            "mean": mean,
            "std": std,
            "num_pixels": num_pixels,
        },
        stats_path
    )

    return mean, std
