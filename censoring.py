import math, cv2, random, torch, torchvision, time
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from scipy.ndimage import gaussian_filter


def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


class abyz22_censoring:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  #  1,H,W,C #image list로 받아서(image batch X)
                "mask1": ("MASK",),  #  1,H,W, # 앞에 1씩 n번 반복임
                "mask2": ("MASK",),  #  1,H,W}
                "sigma": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):  #
        image, mask1, mask2, sigma = kwargs["image"], kwargs["mask1"], kwargs["mask2"], kwargs["sigma"]
        h, w = image.shape[1], image.shape[2]

        if image.shape[1] != mask1.shape[1]:
            mask1 = mask1.permute(0, 2, 1)
        if image.shape[1] != mask2.shape[1]:
            mask2 = mask2.permute(0, 2, 1)
        mask = mask1 + mask2
        if mask.max() < 0.00001:
            return (image,)
        mask_np = np.clip(255.0 * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(mask_np, mode="L")
        # region_mask = self.WT.Masking.smooth_region(pil_image, sigma)

        pil_image = pil_image.convert("L")
        mask_array = np.array(pil_image)
        smoothed_array = gaussian_filter(mask_array, sigma)
        threshold = np.max(smoothed_array) / 2
        smoothed_mask = np.where(smoothed_array >= threshold, 255, 0).astype(np.uint8)
        smoothed_image = Image.fromarray(smoothed_mask, mode="L")
        region_mask = ImageOps.invert(smoothed_image.convert("RGB"))

        region_tensor = pil2mask(region_mask).unsqueeze(0)
        mask = region_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

        image = image.permute(0, 3, 1, 2)
        image_low = torchvision.transforms.Resize((20, 13))(image)
        image_low = torchvision.transforms.Resize((h, w))(image_low)
        image = image.permute(0, 2, 3, 1)
        image_low = image_low.permute(0, 2, 3, 1)

        print(image.shape, image.min(), image.max())
        print(image_low.shape, image_low.min(), image_low.max())
        print(mask.shape, mask.min(), mask.max())
        # image = blend_image(image, image_low, mask)
        image = torch.where(mask > 0.99, image_low, image)

        return (image,)
