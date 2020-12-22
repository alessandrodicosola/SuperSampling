from typing import Tuple

import torch.nn
import math
import torch.nn.functional


def gaussian_filter(kernel_size: int, sigma: float = None):
    """
    Compute gaussian kernel with kernel size and sigma.
    If sigma is None it's automatically computed.

    References
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel
        https://github.com/photosynthesis-team/piq/blob/master/piq/ssim.py
    Args:
        kernel_size: size of the 1D kernel
        sigma: sigma to use

    Returns:
        Tensor (1,kernel_size,kernel_size)
    """
    if kernel_size % 2 == 0:
        raise RuntimeError("Use odd kernel size.")

    if not sigma:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    half = math.floor(kernel_size / 2)
    start = -half
    end = half + 1
    # create the integers where to compute G_mu,sigma(x)
    kernel = torch.arange(start=start, end=end).to(dtype=torch.float32)
    # x ** 2
    kernel = kernel ** 2

    # 1D gaussian has sqrt(2*pi*sigma**2)
    # 2D gaussian has 2*pi*sigma**2
    # left element of the gaussian function
    left = 1 / math.sqrt(2 * math.pi * sigma)
    # numerator of the exponential
    expn = - kernel
    # denominator of the exponential
    expd = 2 * sigma ** 2
    # right element of the gaussian function
    right = torch.exp(expn / expd)
    # 1D kernel
    kernel = left * right
    # kernel has size=kernel_size
    # unsqueeze adds an axis at specified axis
    # create 2D kernel
    kernel = kernel.unsqueeze(1) * kernel.unsqueeze(0)
    # normalize kernel
    kernel = kernel / kernel.sum()
    # return kernel (1,kernel_size,kernel_size)
    return kernel.unsqueeze(0)


class SSIM(torch.nn.Module):
    """Compute SSIM between two images

    References:
        Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
        "Image quality assessment: from error visibility to structural similarity,"
        in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004, doi: 10.1109/TIP.2003.819861.
    """

    def __init__(self, kernel_size_sigma: Tuple[int, float] = (11, 1.5),
                 max_pixel_value: float = 255.,
                 K1_K2: Tuple[float, float] = (0.01, 0.03),
                 alpha_beta_gamma: Tuple[float, float, float] = (1, 1, 1),
                 reduction=None):
        super(SSIM, self).__init__()
        if not reduction:
            raise RuntimeError("Trainer supports only metrics that returns float")
        self.reduction = reduction

        K1, K2 = K1_K2
        self.C1 = (max_pixel_value * K1) ** 2
        self.C2 = (max_pixel_value * K2) ** 2
        self.C3 = self.C2 / 2

        self.alpha, self.beta, self.gamma = alpha_beta_gamma

        kernel_size, sigma = kernel_size_sigma
        self.gaussian_kernel = gaussian_filter(kernel_size=kernel_size, sigma=sigma)

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """

        Args:
            prediction: predicted image
            target: ground truth image

        Returns:
            ssim

        References:
            https://www.cns.nyu.edu/~lcv/ssim/ssim.m
            https://github.com/photosynthesis-team/piq/blob/5f907063f5abe357173a5bed1126b07d46f1b6ac/piq/ssim.py#L350

        """
        if prediction.size() != target.size():
            raise RuntimeError("Size mismatching between prediction and target.")

        rank = len(prediction.size())
        width_axis = rank - 1
        height_axis = width_axis - 1
        channel_axis = height_axis - 1
        channels = prediction.size(channel_axis)
        batch_axis = channel_axis - 1

        if batch_axis == -1:  # inputs are Tensors:
            # insert batch axis
            prediction, target = prediction.unsqueeze(0), target.unsqueeze(0)

        # compute mean ssim over channel dimensions
        ssim_batch = self._compute_ssim_single(prediction, target, channels)

        ssim_batch = ssim_batch.mean(dim=1)

        # compute the ssim over batches
        if self.reduction == "sum":
            return ssim_batch.sum(dim=0).item()
        elif self.reduction == "mean":
            return ssim_batch.mean(dim=0).item()
        else:
            return ssim_batch

    def _compute_ssim_single(self, prediction: torch.Tensor, target: torch.Tensor, channels: int):
        """
        Compute SSIM between two batches of images

        Notes
            Using groups=channels in conv2d allow to apply kernel at each channel independently

        Args:
            prediction: batches of predicted images
            target: batches of ground truth images
            channels: channels in images

        Returns:
            SSIM Tensor (B,C)
        """
        self.gaussian_kernel = self.gaussian_kernel.repeat(channels, 1, 1, 1)

        mu_x = torch.nn.functional.conv2d(input=prediction, weight=self.gaussian_kernel, stride=1, padding=0,
                                          groups=channels)

        mu_y = torch.nn.functional.conv2d(input=target, weight=self.gaussian_kernel, stride=1, padding=0,
                                          groups=channels)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = torch.nn.functional.conv2d(input=prediction * prediction, weight=self.gaussian_kernel, stride=1,
                                                padding=0,
                                                groups=channels) - mu_x_sq
        sigma_y_sq = torch.nn.functional.conv2d(input=target * target, weight=self.gaussian_kernel, stride=1, padding=0,
                                                groups=channels) - mu_y_sq
        sigma_xy = torch.nn.functional.conv2d(input=prediction * target, weight=self.gaussian_kernel, stride=1,
                                              padding=0, groups=channels) - mu_xy
        # with C3 = C2 / 1 and alpha,beta,gamma equals to 1 SSIM contains only luminance and contrast
        # compute mean over spatial dimensions

        return (self._luminance(mu_xy, mu_x_sq, mu_y_sq) * self._contrast(sigma_xy, sigma_x_sq, sigma_y_sq)).mean(
            dim=[-1, -2])

    def _luminance(self, mu_xy, mu_x_sq, mu_y_sq):
        """ Inside the paper they wrote mu_x + mu_y in the matlab implementation there is mu_xy"""
        return (2 * mu_xy + self.C1) / (mu_x_sq + mu_y_sq + self.C1)

    def _contrast(self, sigma_xy, sigma_x_sq, sigma_y_sq):
        return (2 * sigma_xy + self.C2) / (sigma_x_sq + sigma_y_sq + self.C2)

    def _structure(self, sigma_xy, sigma_x, sigma_y):
        return (sigma_xy + self.C3) / (sigma_x * sigma_y + self.C3)
