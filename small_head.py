import math, cv2, random, torch, torchvision, json, os
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드

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


class abyz22_smallhead:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # image= 1,768,512,3
                "segs": ("IMAGE",),  # mask= 1,768,512,3
                "ratio": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 1.5, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "empty_space": (["noise", "black"],),
                "model_detector": ("BOOLEAN", {"default": True, "label_on": "face", "label_off": "head"}),
                "kernel_size": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "dispaly": "slider"}),
                "sigma": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "mask_dilation": ("INT", {"default": 10, "min": 0, "max": 128, "step": 1, "dispaly": "slider"}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "images",
        "masks",
    )

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(sefl, *args, **kwargs):  # image= 1,768,512,3   mask= n,1,h,w
        images, mask = None, None
        print("☆★ " * 10)
        print(len(kwargs["segs"]))
        print(kwargs["segs"][0].shape)
        if len(kwargs["segs"]) == 1:
            kwargs["segs"] = kwargs["segs"][0]
        for i in range(kwargs["image"].shape[0]):
            # y1, x1, y2, x2 = find_white_bbox(kwargs["segs"].permute(0, 3, 1, 2)[i][0].numpy())
            y1, x1, y2, x2 = find_white_bbox(kwargs["segs"][i].permute(2, 0, 1)[0].numpy())
            w = x2 - x1
            h = y2 - y1

            face = kwargs["image"][i].clone()

            face_cropped = face * kwargs["segs"][i]
            if kwargs["empty_space"] == "noise":
                face_noised = torch.rand_like(face) * -1 * (kwargs["segs"] - 1)[i]
            elif kwargs["empty_space"] == "black":
                face_noised = torch.zeros_like(face) * -1 * (kwargs["segs"] - 1)[i]
            face_cropped = face_cropped + face_noised

            face_cropped = face_cropped[y1:y2, x1:x2, :].permute(2, 0, 1)  # 768,512,3 -> 3,768,512
            face_resized = torchvision.transforms.Resize((int(h * kwargs["ratio"]), int(w * kwargs["ratio"])))(face_cropped).permute(
                1, 2, 0
            )  # 3,768,512 -> 768,512,3

            h, w, c = face_resized.shape
            image = kwargs["image"][i]

            image[y1:y2, x1:x2, :] = torch.rand_like(image[y1:y2, x1:x2, :])
            image[y2 - h : y2, int((x1 + x2) // 2 - w // 2) : int((x1 + x2) // 2 - w // 2) + w, :] = face_resized

            # mask 출력
            mask = torch.zeros_like(kwargs["segs"][i, :, :, 0])

            mask_face = kwargs["segs"][i][y1:y2, x1:x2, :].permute(2, 0, 1)
            mask_face = torchvision.transforms.Resize((int(h * kwargs["ratio"]), int(w * kwargs["ratio"])))(mask_face).permute(1, 2, 0)

            h, w, c = mask_face.shape
            mask[y2 - h : y2, int((x1 + x2) // 2 - w // 2) : int((x1 + x2) // 2 - w // 2) + w] = mask_face[:, :, 0]
            mask = -1 * mask + 1
            mask[:y1] = 0
            mask[y2:] = 0
            mask[:, :x1] = 0
            mask[:, x2:] = 0

            kernel = np.ones((kwargs["mask_dilation"], kwargs["mask_dilation"]), np.uint8)
            mask = cv2.dilate(mask.numpy(), kernel, iterations=1)
            mask = torch.tensor(mask)
            mask = mask.unsqueeze(0)

            if kwargs["model_detector"]:
                mask[:, :y1, x1:x2] = 1

            """
            1. 검정배경 생성
            2. 흰색 대가리 작게 생성
            3. 작은 흰색대가리 검정배경에 붙이기
            4. 값 반전 (검정 대가리, 흰배경)
            5. 상하좌우 값 검정으로 만들기
            """

            if i == 0:
                images = image.unsqueeze(0)
                masks = mask
            else:
                images = torch.cat([images, image.unsqueeze(0)])
                masks = torch.cat([masks, mask])

        # mask= n,1,h,w
        masks = torchvision.transforms.GaussianBlur(kernel_size=kwargs["kernel_size"] * 2 + 1, sigma=kwargs["sigma"])(masks)
        if masks.shape[0] ==1:
            masks=masks.unsqueeze(0)
        print(masks.shape)
        return (
            images,
            masks,
        )  # image shoulde be n,768,512,3
