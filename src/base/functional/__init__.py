from functools import partial
import torch.nn.functional

__all__ = ["interpolating_fn"]

interpolating_fn = partial(torch.nn.functional.interpolate,
                           mode="bicubic",
                           align_corners=False,
                           recompute_scale_factor=False)


def extract_patch(image, image_size, patch_size, transform=None):
    """
    Extract patches from an image in an iterator style
    :param image:
    :param image_size: (width,height)
    :param patch_size: (patch_width,patch_height)
    :return: list of ndarray
    """
    image_width, image_height = image_size[0], image_size[1]
    patch_width, patch_height = patch_size[0], patch_size[1]
    col_n_patches = image_width // patch_width + (0 if image_width % patch_width == 0 else 1)
    row_n_patches = image_height // patch_height + (0 if image_height % patch_height == 0 else 1)
    for row in range(row_n_patches):
        for col in range(col_n_patches):
            x = col * patch_width if col * patch_width + patch_width < image_width else image_width - patch_width
            y = row * patch_height if row * patch_height + patch_height < image_height else image_height - patch_height
            # Images are represented as (height,width,channels) in PIL
            patch = image[y:y + patch_height, x:x + patch_width]
            yield patch if not transform else transform(patch)


def restore_image(patches, image_size):
    """
    Restore an image concatenating patches
    :param patches: list of patches
    :param image_size: (width,height,channels)
    :return: image
    """
    import numpy as np
    # Images are represented as (height,width,channels) in PIL
    image = np.zeros((image_size[1], image_size[0], image_size[2]))
    image_width, image_height = image_size[0], image_size[1]
    patch_size = patches[0].shape
    patch_width, patch_height = patch_size[0], patch_size[1]
    # Allow overlapping patches if // is not an integer
    col_n_patches = image_width // patch_width + (0 if image_width % patch_width == 0 else 1)
    row_n_patches = image_height // patch_height + (0 if image_height % patch_height == 0 else 1)
    for i in range(len(patches)):
        col = i % col_n_patches
        row = i // col_n_patches
        x = col * patch_width if col * patch_width + patch_width < image_width else image_width - patch_width
        y = row * patch_height if row * patch_height + patch_height < image_height else image_height - patch_height
        image[y:y + patch_height, x:x + patch_width, :] = patches[i]
    return image
