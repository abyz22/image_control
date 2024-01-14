import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드

# padding1 conditioning 
# padding2 conditioning's'  (와카 여러개 된것 (프롬프트 여러개))

def normalize_size_base_64(w, h):
    short_side = min(w, h)
    remainder = short_side % 64
    return short_side - remainder + (64 if remainder > 0 else 0)


class abyz22_Pad_Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "pad_mode": (["constant", "replicate", "noise"],),
                "mode_type": (
                    [
                        "Top-Left",
                        "Top",
                        "Top-Right",
                        "Center-Left",
                        "Center",
                        "Center-Right",
                        "Bottom-Left",
                        "Bottom",
                        "Bottom-Right",
                        "Random",
                    ],
                ),
                "Ratio_min": ("FLOAT", {"default": 0.5, "min": 0.3, "max": 2.5, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "Ratio_max": ("FLOAT", {"default": 1.5, "min": 0.3, "max": 2.5, "step": 0.1, "round": 0.01, "dispaly": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "CONDITIONING",
        "LATENT",
    )
    RETURN_NAMES = (
        "image",
        "pose_image",
        "conditioning",
        "latent",
    )

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, image, vae, conditioning, control_net_name, pose_strength, mode_type, Ratio_min, Ratio_max, pad_mode,seed):  # image= 1,768,512,3
        if Ratio_min > Ratio_max:
            Ratio_min, Ratio_max = Ratio_max, Ratio_min
        obj = nodes.NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        pose_image = obj.estimate_pose(
            image, "disable", "enable", "disable", resolution=resolution, bbox_detector="yolox_s.onnx", pose_estimator="dw-ss_ucoco.onnx"
        )["result"][0]

        # Resize_by = random.uniform(Ratio_min, Ratio_max)
        np.random.seed(seed)
        random.seed(seed)
        Resize_bys = np.random.uniform(Ratio_min, Ratio_max, image.shape[0]).round(2)
        for i, Resize_by in enumerate(Resize_bys):
            if Resize_by > 0.9999 and Resize_by < 1.0001:
                padded_image = image[i].unsqueeze(0)
                padded_pose_image = pose_image[i].unsqueeze(0)
            else:
                x, y = int(image.shape[2] * Resize_by), int(image.shape[1] * Resize_by)
                resized_image = torchvision.transforms.Resize((y, x))(image[i].permute(2, 0, 1))  # 768,512,3 -> 3,768,512
                resized_pose_image = torchvision.transforms.Resize((y, x), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(
                    pose_image[i].permute(2, 0, 1)
                )  # 1,3,768,512

                # image = n,768,512,3
                # resized_image = 3,768,512
                dx, dy = abs(image.shape[2] - resized_image.shape[2]), abs(image.shape[1] - resized_image.shape[1])
                rdx, rdy = random.randint(0, dx), random.randint(0, dy)
                if Resize_by < 1:
                    mode_list = {
                        "Top-Left": (0, dx, 0, dy),
                        "Top": (int(round(dx / 2)), int(round(dx / 2)) + 1, 0, dy),
                        "Top-Right": (dx, 0, 0, dy),
                        "Center-Left": (0, dx, round(int(dy / 2)), int(round(dy / 2))),
                        "Center": (int(round(dx / 2)), int(round(dx / 2)), int(round(dy / 2)), int(round(dy / 2))),
                        "Center-Right": (dx, 0, int(round(dy / 2)), int(round(dy / 2))),
                        "Bottom-Left": (0, dx, dy, 0),
                        "Bottom": (int(round(dx / 2)), int(round(dx / 2)), dy, 0),
                        "Bottom-Right": (dx, 0, dy, 0),
                        "Random": (rdx, dx - rdx, rdy, dy - rdy),
                    }
                    padded_image = torch.rand_like(image[i].permute(2, 0, 1))  # 3, 768, 512
                    padded_pose_image = torch.zeros_like(pose_image[i].permute(2, 0, 1))

                    if pad_mode == "noise":
                        padded_image[
                            :,
                            mode_list[mode_type][2] : mode_list[mode_type][2] + resized_image.shape[1],
                            mode_list[mode_type][0] : mode_list[mode_type][0] + resized_image.shape[2],
                        ] = resized_image
                        padded_image = padded_image.permute(1, 2, 0).unsqueeze(0)  # 원상복구 768,512,3 후 1,768,512,3
                        padded_pose_image[
                            :,
                            mode_list[mode_type][2] : mode_list[mode_type][2] + resized_image.shape[1],
                            mode_list[mode_type][0] : mode_list[mode_type][0] + resized_image.shape[2],
                        ] = resized_pose_image
                        padded_pose_image = padded_pose_image.permute(1, 2, 0).unsqueeze(0)
                    else:
                        padded_image = torch.nn.functional.pad(resized_image, (mode_list[mode_type]), mode=pad_mode).permute(1, 2, 0).unsqueeze(0)
                        padded_pose_image = (
                            torch.nn.functional.pad(resized_pose_image, (mode_list[mode_type]), mode=pad_mode).permute(1, 2, 0).unsqueeze(0)
                        )
                elif Resize_by > 1:
                    o_h, o_w = image.shape[1], image.shape[2]
                    r_h, r_w = resized_image.shape[1], resized_image.shape[2]
                    mode_list = {
                        "Top-Left": (0, o_w, 0, o_h),
                        "Top": (int(round(dx / 2)), o_w + int(round(dx / 2)), 0, o_h),
                        "Top-Right": (dx, r_w, 0, o_h),
                        "Center-Left": (0, o_w, int(round(dy / 2)), o_h + int(round(dy / 2))),
                        "Center": (int(round(dx / 2)), o_w + int(round(dx / 2)), int(round(dy / 2)), o_h + int(round(dy / 2))),
                        "Center-Right": (dx, r_w, int(round(dy / 2)), o_h + int(round(dy / 2))),
                        "Bottom-Left": (0, o_w, dy, r_h),
                        "Bottom": (int(round(dx / 2)), o_w + int(round(dx / 2)), dy, r_h),
                        "Bottom-Right": (dx, r_w, dy, r_h),
                        "Random": (rdx, o_w + rdx, rdy, o_h + rdy),
                    }
                    padded_image = torch.rand_like(image[0].permute(2, 0, 1))  # 3,768,512
                    padded_pose_image = torch.zeros_like(pose_image[0].permute(2, 0, 1))

                    padded_image = resized_image[
                        :,
                        mode_list[mode_type][2] : mode_list[mode_type][3],
                        mode_list[mode_type][0] : mode_list[mode_type][1],
                    ]
                    padded_pose_image = resized_pose_image[
                        :,
                        mode_list[mode_type][2] : mode_list[mode_type][3],
                        mode_list[mode_type][0] : mode_list[mode_type][1],
                    ]
                    padded_image = padded_image.permute(1, 2, 0).unsqueeze(0)
                    padded_pose_image = padded_pose_image.permute(1, 2, 0).unsqueeze(0)

            if i == 0:
                final_image = padded_image
                final_pose_image = padded_pose_image
            else:
                final_image = torch.cat((final_image, padded_image))
                final_pose_image = torch.cat((final_pose_image, padded_pose_image))

        latent = nodes.VAEEncode().encode(vae, final_image)[0]
        ctrl_net_load = nodes.ControlNetLoader().load_controlnet(control_net_name)[0]
        conditioning1 = nodes.ControlNetApply().apply_controlnet(conditioning, ctrl_net_load, final_pose_image, pose_strength)[0]
        return (
            final_image,
            final_pose_image,
            conditioning1,
            latent,
        )
