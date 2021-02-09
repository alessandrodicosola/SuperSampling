from functools import partial
import torch.nn.functional

__all__ = ["interpolating_fn"]

__interpolating_fn = partial(torch.nn.functional.interpolate,
                             mode="bicubic",
                             align_corners=False,
                             recompute_scale_factor=False)


def interpolating_fn(input, **kwargs):
    return __interpolating_fn(input, **kwargs)
