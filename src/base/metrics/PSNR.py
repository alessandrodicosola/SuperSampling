import torch
import torch.nn

from base.metrics.Metric import Metric


class PSNR(Metric):
    """
    Compute the Peak Signal to Noise Ratio as:

    .. math::

        PSNR = 10 \cdot \log_10 \\frac{MAX_I^2}{\sqrt{MSE_{P,T}}}

    Args:
        dynamic_range: 1 if grayscale, 3 if rgb
        reduction: Reduction applied over the batch: none (not supported by Trainer) | mean | sum
    """

    def __init__(self, dynamic_range: float = 255., reduction=None, denormalize_fn=None):
        super(PSNR, self).__init__(reduction=reduction)
        self.denormalize_fn = denormalize_fn
        self.dynamic_range = dynamic_range
        self.eps = 1e-8

    def compute_values_per_batch(self, *args) -> torch.Tensor:
        prediction, target = args

        if prediction.size() != target.size():
            raise RuntimeError(
                f"Sizes are mismatching: (predictions) {prediction.size()} != (target) {target.size()}")

        prediction = prediction / self.dynamic_range
        target = target / self.dynamic_range

        rank = len(prediction.size())

        # (C,H,W) if r=3 else (B,C,H,W)
        width_dim = rank - 1
        height_dim = width_dim - 1
        channel_dim = height_dim - 1

        mse = torch.mean((prediction - target) ** 2, dim=[channel_dim, height_dim, width_dim])
        psnr = - 10 * torch.log10(mse + self.eps)

        return psnr
