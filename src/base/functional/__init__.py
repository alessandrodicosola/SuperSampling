from functools import partial

import torch.nn.functional

interpolating_fn = partial(torch.nn.functional.interpolate,
                           mode="bicubic",
                           align_corners=False,
                           recompute_scale_factor=False)