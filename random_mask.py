import math, cv2, random, torch, torchvision, time
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드
from PIL import Image, ImageDraw, ImageFilter
from typing import Union, List


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]


class abyz22_RandomMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "multi_mask": ("BOOLEAN", {"default": True, "label_on": "multi", "label_off": "single"}),
                "shape": (
                    ["circle", "square", "triangle", "random"],
                    {"default": "circle"},
                ),
                "size": (
                    ["small", "medium", "big", "random"],
                    {"default": "medium"},
                ),
                "mask_min": ("FLOAT", {"default": 0, "min": -0.1, "max": 1, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "mask_max": ("FLOAT", {"default": 1, "min": 0, "max": 1.1, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "expand": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "blur": (
                    ["very_weak", "weak", "medium", "hard", "very_hard", "random"],
                    {"default": "medium"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {},
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )  # image = N, H, W, C
    RETURN_NAMES = (
        "image",
        "mask",
    )  #   mask= N, H, W

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):
        image = kwargs["image"]
        torch.manual_seed(time.time())
        random.seed(time.time())
        loc_x, loc_y = random.randint(0, image.shape[2]), random.randint(0, image.shape[1])
        shape, size, mask_min, mask_max, blur, expand = (
            kwargs["shape"],
            kwargs["size"],
            kwargs["mask_min"],
            kwargs["mask_max"],
            kwargs["blur"],
            kwargs["expand"],
        )

        if size == "small":
            size_list = [0.2, 0.3]
        elif size == "medium":
            size_list = [0.35, 0.45]
        elif size == "big":
            size_list = [0.5, 0.6]
        elif size == "random":
            size_list = [0.2, 0.5]
        batch_num = image.shape[0] if kwargs["multi_mask"] else 1
        masks = None
        for i in range(batch_num):
            mask = np.zeros((image.shape[1], image.shape[2]))

            w = random.randint(int(image.shape[2] * size_list[0]), int(image.shape[2] * size_list[1])) // 2
            h = random.randint(int(image.shape[1] * size_list[0]), int(image.shape[1] * size_list[1])) // 2
            random_value = random.random()

            if shape == "circle" or (shape == "random" and random_value < 0.33):
                cv2.circle(mask, (loc_x, loc_y), w, (255), -1)
            elif shape == "square" or (shape == "random" and random_value < 0.66):
                cv2.rectangle(mask, (loc_x - w, loc_y - h), (loc_x + w, loc_y + h), (255), -1)
            elif shape == "triangle" or (shape == "random" and random_value < 1):
                pt1 = (int(loc_x - w // 2), int(loc_y + h // 2))  # bottom left
                pt2 = (int(loc_x + w // 2), int(loc_y + h // 2))  # bottom right
                pt3 = (int(loc_x), int(loc_y - h // 2))  # top point
                pts = np.array([pt1, pt2, pt3])
                cv2.fillPoly(mask, [pts], True, 255)

            if shape == "random":
                m = cv2.getRotationMatrix2D((image.shape[2] / 2, image.shape[1] / 2), random.randint(0, 360), 1)
                mask = cv2.warpAffine(mask, m, (image.shape[2], image.shape[1]))

            mask = np.expand_dims(mask, 0)
            if masks is None:
                masks = mask
            else:
                masks = np.concatenate([masks, mask], axis=0)

        masks = torch.Tensor(masks)

        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        growmask = masks.reshape((-1, masks.shape[-2], masks.shape[-1])).cpu()
        out = []
        current_expand = expand
        for m in growmask:
            output = m.numpy()
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    # output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                    output = cv2.erode(output, kernel)
                else:
                    # output = scipy.ndimage.grey_dilation(output, footprint=kernel)
                    output = cv2.dilate(output, kernel)
            output = torch.from_numpy(output)

            out.append(output)
        if blur == "very_weak":
            blur_radius = random.randint(1, 15)
        elif blur == "weak":
            blur_radius = random.randint(15, 30)
        elif blur == "medium":
            blur_radius = random.randint(30, 50)
        elif blur == "hard":
            blur_radius = random.randint(50, 80)
        elif blur == "very_hard":
            blur_radius = random.randint(80, 120)
        elif blur == "random":
            blur_radius = random.randint(1, 120)

        # Convert the tensor list to PIL images, apply blur, and convert back
        for idx, tensor in enumerate(out):
            # Convert tensor to PIL image
            pil_image = tensor2pil(tensor.cpu().detach())[0]
            # Apply Gaussian blur
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
            # Convert back to tensor
            out[idx] = pil2tensor(pil_image)

        if mask_max > 1:
            rand = max(min((torch.randn(1) + 0.75) / 2, 1), 0)
            mask_max = max(rand, 0.5)
        if mask_min < 0:
            rand = max(min((torch.randn(1) + 0.35) / 2, 1), 0)
            mask_min = min(rand, 0.5)
        masks = torch.cat(out, dim=0)
        # 값 증폭(blur 높으면 밝은 쪽이 사라짐)

        mask = mask * 3
        masks = torch.where(masks > mask_max, mask_max, masks)
        masks = torch.where(masks < mask_min, mask_min, masks)
        images = masks.unsqueeze(-1).expand(-1, -1, -1, 3)
        return (
            images,
            masks,
        )


class abyz22_AddPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "weight_factor": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 1.5, "step": 0.05}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "red light/blue light/yellow light/sun light/pink light/blue light/green light/purple light/warm atmosphere/sunshine from window/sunshine, outdoor, warm atmosphere/shadow from window/sunset over sea/neon light, city,cyberpunk/light and shadow/sci-fi RGB glowing, cyberpunk/natural lighting/shadow from window/sunset over sea/sci-fi RGB glowing, studio lighting/",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):
        c0, text, clip, weight_factor = kwargs["conditioning"], kwargs["text"], kwargs["clip"], kwargs["weight_factor"]
        random.seed(time.time())

        if text.endswith("/"):
            text = text[:-1]
        text = random.choice(text.split("/"))
        text = f"({text}:{weight_factor})"

        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c1 = [[cond, {"pooled_output": pooled}]]
        out = []

        if len(c0) > 1:
            print("☆★" * 20)

        cond_from = c0[0][0]

        for i in range(len(c1)):
            t1 = c1[i][0]
            tw = torch.cat((t1, cond_from), 1)
            n = [tw, c1[i][1].copy()]
            out.append(n)
        return (out,)
