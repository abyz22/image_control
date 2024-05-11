import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드


class abyz22_random_mask:
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
                "min": ("FLOAT", {"default": 0, "min": -0.1, "max": 1, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "max": ("FLOAT", {"default": 1, "min": 0, "max": 1.1, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "edge": (
                    ["very_soft", "soft", "medium", "hard", "very_hard", "random"],
                    {"default": "medium"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "MASK")  # image = N, H, W, C
    RETURN_NAMES = ("image", "mask")  #   mask= N, H, W

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):
        image = kwargs["image"]
        loc_x, loc_y = random.randint(0, image.shape[2]), random.randint(0, image.shape[1])
        shape, size, min, max, edge = kwargs["shape"], kwargs["size"], kwargs["min"], kwargs["max"], kwargs["edge"]

        if size == "small":
            size_list = [0, 0.2]
        elif size == "medium":
            size_list = [0.4, 0.6]
        elif size == "big":
            size_list = [0.6, 0.8]
        elif size == "random":
            size_list = [0, 0.8]
        batch_num = image.shape[0] if kwargs["multi_mask"] else 1
        masks = []
        for i in range(batch_num):
            mask = np.zeros((image.shape[1], image.shape[2]))

            w = random.randint(image.shape[2] * size_list[0], image.shape[2] * size_list[1]) // 2
            h = random.randint(image.shape[1] * size_list[0], image.shape[1] * size_list[1]) // 2
            if shape == "circle" or shape == "square" or (shape == "random" and random.random() < 0.3):
                if shape == "circle":
                    draw_poly = cv2.circle
                elif shape == "sqaure":
                    draw_poly = cv2.rectangle
                draw_poly(mask, (loc_x - w, loc_y - h), (loc_x + w, loc_y + h))
            elif shape == "triangle":

                pt1 = (loc_x - w // 2, loc_y + h // 2)  # bottom left
                pt2 = (loc_x + w // 2, loc_y + h // 2)  # bottom right
                pt3 = (loc_x, loc_y - h // 2)  # top point
                pts = np.array([pt1, pt2, pt3])
                cv2.polylines(mask, [pts], True, 255, -1)
            if shape == "random":
                m = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[2] / 2), random.randint(0, 360), 1)
                img = cv2.warpAffine(img, m, (image.shape[2], image.shape[1]))

            # TODO IMAGE 블러

            mask = np.expand_dims(mask, 0)
            if masks == []:
                masks = mask
            else:
                masks = np.concatenate([masks, mask], axis=0)
        return masks
