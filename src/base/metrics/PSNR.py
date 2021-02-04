import torch
import torch.nn

from base.metrics.Metric import Metric


class PSNR(Metric):
    """
    Compute the Peak Signal to Noise Ratio as:

    .. math::

        PSNR = 10 \cdot \log_10 \\frac{MAX_I^2}{\sqrt{MSE_{P,T}}}

    Args:
        max_pixel_value: 1 if grayscale, 3 if rgb
        reduction: Reduction applied over the batch: none (not supported by Trainer) | mean | sum
    """

    def __init__(self, max_pixel_value: float = 255., reduction=None, denormalize_fn=None):
        super(PSNR, self).__init__(reduction=reduction)
        self.denormalize_fn = denormalize_fn
        self.max_pixel_value = max_pixel_value
        self.eps = 1e-8

    def compute_values_per_batch(self, *args) -> torch.Tensor:
        prediction, target = args

        if prediction.size() != target.size():
            raise RuntimeError(
                f"Sizes are mismatching: (predictions) {prediction.size()} != (target) {target.size()}")

        prediction = prediction / self.max_pixel_value if not self.denormalize_fn else self.denormalize_fn(
            prediction) / self.max_pixel_value
        target = target / self.max_pixel_value if not self.denormalize_fn else self.denormalize_fn(
            target) / self.max_pixel_value

        rank = len(prediction.size())

        # (C,H,W) if r=3 else (B,C,H,W)
        width_dim = rank - 1
        height_dim = width_dim - 1
        channel_dim = height_dim - 1
        batch_dim = channel_dim - 1

        mse = torch.mean((prediction - target) ** 2, dim=[channel_dim, height_dim, width_dim])
        mse = mse + self.eps
        # if mse is close to 0 returns 100 otherwise returns psnr
        psnr = torch.where(torch.isclose(mse, torch.zeros_like(mse)),
                           100 * torch.ones_like(mse),
                           - 10 * torch.log10(mse))

        return psnr
