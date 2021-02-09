import math
from typing import Tuple

import torch
import torch.nn
import torch.nn.functional

from base.metrics.Metric import Metric


def gaussian_filter(kernel_size: int, sigma: float = None) -> torch.Tensor:
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


class SSIM(Metric):
    """Compute SSIM between two images.

    Notes
        No conversion is applied if RGB images are used: it's computd the mean over the channels.

    References:
        Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
        "Image quality assessment: from error visibility to structural similarity,"
        in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004, doi: 10.1109/TIP.2003.819861.

    Args:
        kernel_size_sigma: Tuple (kernel_size, sigma) for initialized gaussian kernel. Default: (11,1.5)
        dynamic_range: Difference between max and min pixel values.
        image_channels: Channels of the input images.
        K1_K2: Tuple(float,float) for creating constants C1,C2. Default: (0.01, 0.03)
        alph_beta_gamma: Exponents for computing ssim with luminance, contrast, structure. Default: (1,1,1)
        reduction: Reduction applied to the final ssim. Default: None (return a Tensor [BATCH,SSIM]). Possible values: {mean, sum}.
        denormalize_fn: None ?!
    """

    def __init__(self, kernel_size_sigma: Tuple[int, float] = (11, 1.5), dynamic_range: float = 255., image_channels=3,
                 K1_K2: Tuple[float, float] = (0.01, 0.03), alpha_beta_gamma: Tuple[float, float, float] = (1, 1, 1),
                 reduction=None):
        super(SSIM, self).__init__(reduction=reduction)
        if not reduction:
            raise RuntimeError(
                "Trainer supports only metrics that returns float: .items() at the end. Use reduction={mean, sum}")
        self.dynamic_range = dynamic_range

        self.reduction = reduction

        K1, K2 = K1_K2
        self.C1 = (dynamic_range * K1) ** 2
        self.C2 = (dynamic_range * K2) ** 2
        self.C3 = self.C2 / 2

        self.alpha, self.beta, self.gamma = alpha_beta_gamma

        kernel_size, sigma = kernel_size_sigma

        # register the gaussian kernel in order to be moved to gpu when .to is called
        self.register_buffer('gaussian_kernel',
                             gaussian_filter(kernel_size=kernel_size, sigma=sigma).repeat(image_channels, 1, 1, 1),
                             persistent=True)

    def compute_values_per_batch(self, *args: torch.Tensor):
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
        prediction, target = args
        prediction = prediction / self.dynamic_range
        target = target / self.dynamic_range

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

        ssim_batch = self._compute_ssim_single(prediction, target, channels)
        # compute mean ssim over channel dimensions
        ssim_batch = ssim_batch.mean(dim=1)

        return ssim_batch

    def _compute_ssim_single(self, prediction: torch.Tensor, target: torch.Tensor, channels: int):
        """
        Compute SSIM between two batches of images

        Notes
            Using groups=channels in conv2d allow to apply kernel at each channel independently
            @torchno_grad has safety measure
        Args:
            prediction: batches of predicted images
            target: batches of ground truth images
            channels: channels in images

        Returns:
            SSIM Tensor (B,C)
        """
        mu_x = torch.nn.functional.conv2d(input=prediction, weight=self.gaussian_kernel, stride=1, padding=0,
                                          groups=channels)

        mu_y = torch.nn.functional.conv2d(input=target, weight=self.gaussian_kernel, stride=1, padding=0,
                                          groups=channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        del mu_x
        del mu_y

        sigma_x_sq = torch.nn.functional.conv2d(input=prediction * prediction, weight=self.gaussian_kernel, stride=1,
                                                padding=0,
                                                groups=channels) - mu_x_sq
        sigma_y_sq = torch.nn.functional.conv2d(input=target * target, weight=self.gaussian_kernel, stride=1, padding=0,
                                                groups=channels) - mu_y_sq
        sigma_xy = torch.nn.functional.conv2d(input=prediction * target, weight=self.gaussian_kernel, stride=1,
                                              padding=0, groups=channels) - mu_xy

        # with C3 = C2 / 1 and alpha,beta,gamma equals to 1 SSIM contains only luminance and contrast
        left = self._ssim_simplified_1(mu_xy, mu_x_sq, mu_y_sq)

        del mu_x_sq
        del mu_y_sq
        del mu_xy

        right = self._ssim_simplified_2(sigma_xy, sigma_x_sq, sigma_y_sq)

        # compute mean over spatial dimensions
        return (left * right).mean(dim=[-1, -2])

    def _ssim_simplified_1(self, mu_xy, mu_x_sq, mu_y_sq):
        return (2 * mu_xy + self.C1) / (mu_x_sq + mu_y_sq + self.C1)

    def _ssim_simplified_2(self, sigma_xy, sigma_x_sq, sigma_y_sq):
        return (2 * sigma_xy + self.C2) / (sigma_x_sq + sigma_y_sq + self.C2)

    def _luminance(self, mu_x, mu_y):
        return (2 * mu_x * mu_y + self.C1) / (torch.pow(mu_x, 2) + torch.pow(mu_y, 2) + self.C1)

    def _contrast(self, sigma_x, sigma_y):
        return (2 * sigma_x * sigma_y + self.C2) / (torch.pow(sigma_x, 2) + torch.pow(sigma_y, 2) + self.C2)

    def _structure(self, sigma_xy, sigma_x, sigma_y):
        return (sigma_xy + self.C3) / (sigma_x * sigma_y + self.C3)

    def _ssim(self, mu_x, mu_y, sigma_x, sigma_y, sigma_xy):
        return \
            torch.pow(self._luminance(mu_x, mu_y), self.alpha) + \
            torch.pow(self._contrast(sigma_x, sigma_y), self.beta) + \
            torch.pow(self._structure(sigma_xy, sigma_x, sigma_y), self.gamma)
