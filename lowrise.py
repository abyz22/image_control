import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드
from PIL import Image
from PIL import ImageDraw

# class name:
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#         "required": {},
#         "optional": {},
#         }
#     RETURN_TYPES = ()
#     RETURN_NAMES = ()

#     FUNCTION = "run"

#     CATEGORY = "abyz22"

#     def run(sefl,*args,**kwargs):
#         return None


def find_white_bbox(mask):
    mask = (mask > 0).astype(np.uint8)
    # 행과 열의 합을 계산하여 흰색 영역을 찾습니다.
    row_sum = np.sum(mask, axis=1)  # 768
    col_sum = np.sum(mask, axis=0)  # 512

    # 흰색 영역의 시작과 끝 인덱스를 찾습니다.
    for i, x in enumerate(row_sum):
        if x > 0:
            break
    for ii, x in enumerate(col_sum):
        if x > 0:
            break
    y1, x1 = i, ii

    for i, x in enumerate(row_sum[::-1]):
        if x > 0:
            break
    for ii, x in enumerate(col_sum[::-1]):
        if x > 0:
            break
    y2, x2 = row_sum.shape[0] - i, col_sum.shape[0] - ii
    return y1, x1, y2, x2


class abyz22_drawmask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
                "dx": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "dy": ("FLOAT", {"default": 0.5, "min": 0, "max": 2.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "dy2": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "dy3": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "mode_type": (
                    [
                        "Waist-detect",
                        "pussy-detect",
                    ],
                ),
            },
            "optional": {"person_SEGS": ("SEGS",)},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(sefl, *args, **kwargs):
        obj = nodes.NODE_CLASS_MAPPINGS["ImpactSEGSToMaskBatch"]()
        mask = obj.doit(kwargs["segs"])[0]
        if torch.all(mask[0] == 0):
            return (mask,)
        y1, x1, y2, x2 = find_white_bbox(mask[0].numpy())  # x1,y1,x2,y2
        w = x2 - x1
        h = y2 - y1

        dx = int(w * kwargs["dx"])
        dy = int(h * kwargs["dy"])
        dy2 = y2 - int(h * kwargs["dy2"])
        if kwargs["mode_type"] == "Waist-detect":
            mask = mask.numpy()[0]
            mask[: int((y1 + y2) // 2)] = 0

            cv2.ellipse(mask, ((x1 + x2) // 2, int(y2 * 0.97)), (dx, dy), 0, 0, 180, 1, -1)

            cv2.rectangle(mask, (x1, int(y2 * 0.95)), (x2, dy2), 1, -1)
        elif kwargs["mode_type"] == "pussy-detect":
            mask = mask.numpy()[0]

            person_mask = obj.doit(kwargs["person_SEGS"])[0]

            person_mask = person_mask.numpy()[0]
            person_mask[: int(y1 - kwargs["dy2"] * h), :] = 0  # 윗부분 자르기
            person_mask[int(y1 - kwargs["dy3"] * h) :, :] = 0  # 아랫부분 자르기

            cv2.ellipse(person_mask, (int((x1 + x2) // 2), int(y1 - kwargs["dy3"] * h)), (int(dx), int(dy)), 0, 0, 180, 1, -1) #아랫부분 끝부터

            # cv2.rectangle(person_mask, (int(x1 - 3 * w), int(y1 - 2 * h)), (int(x2 + 3 * w), int(y2)), 1, -1)
            mask = person_mask

        mask = np.array(mask)
        mask = torch.tensor(mask).unsqueeze(0)

        return (mask,)


class abyz22_blendimages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "imageB": ("IMAGE",),
                "mask": ("IMAGE",),
                "blend_percentage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(sefl, *args, **kwargs):
        obj = nodes.NODE_CLASS_MAPPINGS["Image Blend by Mask"]()
        images = None
        for i in range(kwargs["imageA"].shape[0]):
            image = obj.image_blend_mask(
                kwargs["imageA"][i].unsqueeze(0), kwargs["imageB"][i].unsqueeze(0), kwargs["mask"][i].unsqueeze(0), kwargs["blend_percentage"]
            )[0]
            if i == 0:
                images = image
            else:
                images = torch.cat([images, image])
        return (images,)


class abyz22_blend_onecolor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "mask": ("IMAGE",),
                "blend_percentage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "dispaly": "slider"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "dispaly": "slider"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "dispaly": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(sefl, *args, **kwargs):
        obj = nodes.NODE_CLASS_MAPPINGS["Image Blend by Mask"]()

        # one_color = torch.zeros_like(kwargs["imageA"])
        one_color = torch.ones_like(kwargs["imageA"])

        one_color[:, :, :, 0].fill_(kwargs["R"] / 255.0)
        one_color[:, :, :, 1].fill_(kwargs["G"] / 255.0)
        one_color[:, :, :, 2].fill_(kwargs["B"] / 255.0)

        for i in range(kwargs["imageA"].shape[0]):
            image = obj.image_blend_mask(
                kwargs["imageA"][i].unsqueeze(0), one_color[i].unsqueeze(0), kwargs["mask"][i].unsqueeze(0), kwargs["blend_percentage"]
            )[0]
            if i == 0:
                images = image
            else:
                images = torch.cat([images, image])
        return (images,)
