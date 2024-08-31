import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드


class abyz22_path_generator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_path": ("STRING", {"default": ""}),
                "start_num": ("INT", {"default": 1, "min": 1, "max": 3000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):
        basic_path, start_num = kwargs["basic_path"], kwargs["start_num"]

        t = basic_path + str(start_num)
        return (t,)
