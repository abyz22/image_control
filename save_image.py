import math, cv2, random, torch, torchvision, json, os
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import nodes, folder_paths  # 기본노드, 파일로드


class abyz22_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_path": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "abyz22"

    def save_images(self, images, folder_path, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        _, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )


        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        full_output_folder = folder_path

        file_list = os.listdir(folder_path)
        file_list= [int(f.split('.')[0]) for f in file_list]
        file_list.sort(reverse=True)
        counter = 1 if len(file_list)==0 else file_list[0] +1

        results = list()
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{counter}.png"
            if folder_path != 'no-data':
                img.save(os.path.join(full_output_folder, file), pnginfo=None, compress_level=self.compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}
