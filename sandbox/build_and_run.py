import numpy as np
import PIL
import torch
import einops as eo
import torchvision.transforms.functional as tvf

from segment_anything.build_sam import build_sam_vit_b
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


def load_image(path, target_size=(1024, 1024), shape_format: str = "c h w"):
    image = PIL.Image.open(path)
    image = image.convert("RGB")
    image = torch.from_numpy(np.asarray(image))
    image = eo.rearrange(image, "h w c -> c h w")
    image = tvf.resize(image, size=target_size)
    if shape_format == "c h w":
        return image
    elif shape_format == "h w c":
        return eo.rearrange(image, "c h w -> h w c")
    else:
        raise ValueError(f"Unknown shape format {shape_format}")


def single_point_run(image_path, checkpoint_path):
    sam = build_sam_vit_b(checkpoint=checkpoint_path)

    image = load_image(image_path)

    sam_input = {
        "image": image,
        "original_size": tuple(image.shape[:2]),
        "point_coords": torch.tensor([p / 2 for p in image.shape[:2]]).reshape(1, 1, 2),
        "point_labels": torch.zeros(1, 1),
    }

    sam_output = sam([sam_input], multimask_output=True)
    print(sam_output)


def create_image_with_center_patch(size=256, target_size=(1024, 1024)) -> np.ndarray:
    H, W = target_size

    # Create a zero array of shape (H, W)
    img = np.zeros((H, W))

    # Compute the coordinates of the central patch
    x0 = W // 2 - size // 2
    x1 = x0 + size
    y0 = H // 2 - size // 2
    y1 = y0 + size

    # Set the central patch to ones
    img[y0:y1, x0:x1] = 1

    return img


def auto_mask_generator_run(image_path, checkpoint_path):
    sam = build_sam_vit_b(checkpoint=checkpoint_path)

    image = load_image(image_path, shape_format="h w c")
    image = image.numpy()

    local_score_bias = create_image_with_center_patch(
        size=256, target_size=image.shape[:2]
    )

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        min_local_score_thresh_for_point_skip=0.5,
    )
    sam_output = mask_generator.generate(image, local_score_bias=local_score_bias)
    print(sam_output)


if __name__ == "__main__":
    image_path_ = "/Users/nrahaman/Python/segment-anything/test_images/airport.png"
    checkpoint_path_ = (
        "/Users/nrahaman/Python/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    )

    auto_mask_generator_run(image_path_, checkpoint_path_)
