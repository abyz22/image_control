import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths, comfy
from . import wildcards


class abyz22_ImpactWildcardEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
                "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "Populate", "label_off": "Fixed"}),
                "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    CATEGORY = "abyz22"

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONINGS", "STRING")
    RETURN_NAMES = ("model", "clip", "conditionings", "populated_text")
    FUNCTION = "doit"

    @staticmethod
    def process_with_loras(**kwargs):
        return wildcards.process_with_loras(**kwargs)

    @staticmethod
    def get_wildcard_list():  # 이건 사용 안함 (왜있는지 모름)
        return wildcards.get_wildcard_list()

    def doit(self, *args, **kwargs):
        wildcard_process = nodes.NODE_CLASS_MAPPINGS["ImpactWildcardProcessor"].process

        conditioning = []
        for i in range(kwargs["batch_size"]):
            populated = wildcard_process(text=kwargs["wildcard_text"], seed=kwargs["seed"] + i)
            # model, clip, conditioning = wildcards.process_with_loras(populated, kwargs["model"], kwargs["clip"])
            model, clip, con = wildcards.process_with_loras(populated, kwargs["model"], kwargs["clip"])
            conditioning.append(con[0])
        conditionings = [conditioning[i : i + 1] for i in range(len(conditioning))]  # 앞에 차원 1추가한뒤 쌓기
        # print("x" * 30)

        # conditioning = conds[0]
        # conditioning[0][0] = torch.cat((conditioning[0][0]), conds[0], 0)
        # model, clip, conditioning = wildcards.process_with_loras(populated, kwargs["model"], kwargs["clip"])
        # print("len conditioning ", len(conditioning))  # 1
        # print("len conditioning[0] ", len(conditioning[0]))  # 2  ('pooled_output이 섞여있음)
        # print("len conditioning[0][0] ", len(conditioning[0][0]))
        # print("shape conditioning[0][0] ", conditioning[0][0].shape)  # 1, 77, 768
        # print(conditioning[0])

        return (model, clip, conditionings, populated)


class abyz22_KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positives": ("CONDITIONINGS",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vae_decode": (["true", "true (tiled)", "false"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
                "script": ("SCRIPT",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONINGS",
        "CONDITIONING",
        "LATENT",
        "VAE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "MODEL",
        "CONDITIONINGS+",
        "CONDITIONING-",
        "LATENT",
        "VAE",
        "IMAGE",
    )
    FUNCTION = "doit"
    CATEGORY = "abyz22"

    def doit(self, *args, **kwargs):
        pos=kwargs["positives"]
        ksample_latents = {}

        obj = nodes.NODE_CLASS_MAPPINGS["KSampler"]()
        for i, positive in enumerate(kwargs["positives"]):
            kwargs["latent_image"]["samples"] = kwargs["latent_image"]["samples"][0:1, :, :, :]
            ksampler_result = obj.sample(
                model=kwargs["model"],
                seed=kwargs["seed"] + i,
                steps=kwargs["steps"],
                cfg=kwargs["cfg"],
                sampler_name=kwargs["sampler_name"],
                scheduler=kwargs["scheduler"],
                positive=positive,
                negative=kwargs["negative"],
                latent_image=kwargs["latent_image"],
                denoise=kwargs["denoise"],
            )
            ksample_latent = ksampler_result[0]
            if i == 0:
                ksample_latents["samples"] = ksample_latent["samples"]
            if i > 0:
                ksample_latents["samples"] = torch.cat((ksample_latents["samples"], ksample_latent["samples"]), 0)

        obj2 = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]()
        images = obj2.decode(kwargs["optional_vae"], ksample_latents)[0]

        return (kwargs["model"], kwargs["positives"], kwargs["negative"], kwargs["latent_image"], kwargs["optional_vae"], images)


class Pad_Image_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "conditionings": ("CONDITIONINGS",),
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
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "CONDITIONINGS",
        "LATENT",
    )
    RETURN_NAMES = (
        "image",
        "pose_image",
        "conditionings",
        "latent",
    )

    FUNCTION = "doit"

    CATEGORY = "abyz22"

    def doit(self, *args, **kwargs):
        image, vae, control_net_name, pose_strength, mode_type, Ratio_min, Ratio_max, pad_mode = (
            kwargs["image"],
            kwargs["vae"],
            kwargs["control_net_name"],
            kwargs["pose_strength"],
            kwargs["mode_type"],
            kwargs["Ratio_min"],
            kwargs["Ratio_max"],
            kwargs["pad_mode"],
        )
        # image= 1,768,512,3
        if Ratio_min > Ratio_max:
            Ratio_min, Ratio_max = Ratio_max, Ratio_min
        obj = nodes.NODE_CLASS_MAPPINGS["DWPreprocessor"]()

        def normalize_size_base_64(w, h):
            short_side = min(w, h)
            remainder = short_side % 64
            return short_side - remainder + (64 if remainder > 0 else 0)

        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        pose_image = obj.estimate_pose(
            image, "disable", "enable", "disable", resolution=resolution, bbox_detector="yolox_s.onnx", pose_estimator="dw-ss_ucoco.onnx"
        )["result"][0]

        # Resize_by = random.uniform(Ratio_min, Ratio_max)
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
        ##############################

        same_count = 1
        positives = []

        # opt1
        print('☆★'*20)
        print('☆★'*20)
        for i, positive in enumerate(kwargs["conditionings"]):
            if i < len(kwargs["conditionings"]) - 1:
                if np.array_equal(
                    positive[0][0].cpu().detach().numpy(), kwargs["conditionings"][i + 1][0][0].cpu().detach().numpy()
                ) and np.array_equal(
                    positive[0][1]["pooled_output"].cpu().detach().numpy(),
                    kwargs["conditionings"][i + 1][0][1]["pooled_output"].cpu().detach().numpy(),
                ):  # 현재거랑 다음거랑 같으면
                    same_count += 1
                    continue
            ctrl_net_load = nodes.ControlNetLoader().load_controlnet(control_net_name)[0]
            positive = nodes.ControlNetApply().apply_controlnet(positive, ctrl_net_load, final_pose_image[i, i + same_count], pose_strength)[0]
            print(i)
            print(positive[0][1]['control'])

            for ii in range(same_count):
                positives.append(positive)
            same_count = 1
        print('☆★'*20)
        print('☆★'*20)

        #opt2
        # ctrl_net_load = nodes.ControlNetLoader().load_controlnet(control_net_name)[0]
        # positives = nodes.ControlNetApply().apply_controlnet(kwargs["conditionings"][0], ctrl_net_load, final_pose_image, pose_strength)[0]
        # positives=[positives]

        # print(len(positives))
        # print(len(positives[0]))
        # print(len(positives[0][0]))
        # print(len(positives[0][0][0]))
        # print(positives[0][0][0].shape)

        return (
            final_image,
            final_pose_image,
            positives,
            latent,
        )


class abyz22_ToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positives": ("CONDITIONINGS",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    FUNCTION = "doit"

    CATEGORY = "abyz22"

    def doit(self, model, clip, vae, positives, negative):
        pipe = (model, clip, vae, positives, negative)
        return (pipe,)


class abyz22_FromBasicPipe_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
            },
        }

    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONINGS", "CONDITIONING")
    RETURN_NAMES = ("basic_pipe", "model", "clip", "vae", "positives", "negative")
    FUNCTION = "doit"
    CATEGORY = "abyz22"

    def doit(self, basic_pipe):
        model, clip, vae, positives, negative = basic_pipe
        return basic_pipe, model, clip, vae, positives, negative


# class abyz22_Ultimate_SD_Upscale:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {}
#     RETURN_TYPES = ()
#     RETURN_NAMES=()

#     FUNCTION = "doit"
#     CATEGORY = "abyz22"

#     def doit(self, *args, **kwargs):
#         return None

# class abyz22_DetailerDebug_Segs:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {}
#     RETURN_TYPES = ()
#     RETURN_NAMES=()

#     FUNCTION = "doit"
#     CATEGORY = "abyz22"

#     def doit(self, *args, **kwargs):
#         return None
