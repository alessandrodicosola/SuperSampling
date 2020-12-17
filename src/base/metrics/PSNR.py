import torch
import torch.nn
import math


class PSNR(torch.nn.Module):
    """
    Compute the Peak Signal to Noise Ratio as:

    .. math::

        PSNR = 10 \cdot \log_10 \\frac{MAX_I^2}{\sqrt{MSE_{P,T}}}

    """

    def __init__(self, max_pixel_value=255):
        super(PSNR, self).__init__()
        self.max_pixel_value = max_pixel_value
        self.numerator = 20 * math.log10(max_pixel_value)
        self.eps = 1e-8

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        if prediction.size() != target.size():
            raise RuntimeError(
                f"Sizes are mismatching: (predictions) {prediction.size()} != (target) {target.size()}")

        # scale: now max value is 1
        prediction = prediction / self.max_pixel_value
        target = target / self.max_pixel_value

        mse = ((prediction - target) ** 2).view(-1, 1).mean(dim=1).sum()
        if mse == 0:
            return 100

        psnr = 10 * torch.log10(1 / mse + self.eps)
        psnr = psnr.item()
        return psnr
