import math

import cv2

# https://github.com/advimman/lama
import logging

import torch, torchvision, random

import numpy as np
import os
import sys
from PIL import Image
from numpy import dtype
from scipy import ndimage
from torch import device

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, ".."))
comfy_dir = os.path.abspath(os.path.join(my_dir, "..", ".."))

# Construct the path to the font file
font_path = os.path.join(my_dir, "arial.ttf")

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
from nodes import (
    LatentUpscaleBy,
    KSampler,
    KSamplerAdvanced,
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeTiled,
    ImageScaleBy,
    CLIPSetLastLayer,
    CLIPTextEncode,
    ControlNetLoader,
    ControlNetApply,
    ControlNetApplyAdvanced,
    SetLatentNoiseMask,
    LoadImageMask,
)
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats

# Ottieni il percorso assoluto alla radice del tuo progetto
percorso_radice_progetto = os.path.abspath(os.path.dirname(__file__))

# Aggiungi il percorso alla variabile d'ambiente PYTHONPATH
sys.path.append(percorso_radice_progetto)

LOGGER = logging.getLogger(__name__)

from annotator.lama import LamaInpainting


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


model_lama = None


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


import numpy as np
import torch
from einops import rearrange
import cv2


def safe_numpy(x):
    y = x.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y


def get_pytorch_control(x):
    y = torch.from_numpy(x)
    y = y.float() / 255.0
    y = rearrange(y, "h w c -> 1 c h w")
    y = y.clone()
    y = y.to("cuda" if torch.cuda.is_available() else "cpu")  # Assumendo che tu abbia definito la variabile 'devices'
    y = y.clone()
    return y


def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]


def high_quality_resize(x, size):
    # Written by lvmin
    # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

    # Verifica se l'immagine ha un canale alpha e, in caso affermativo, separalo
    inpaint_mask = None
    if x.ndim == 3 and x.shape[2] == 4:
        inpaint_mask = x[:, :, 3]
        x = x[:, :, 0:3]

    new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
    new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
    unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
    is_one_pixel_edge = False
    is_binary = False
    if unique_color_count == 2:
        is_binary = np.min(x) < 16 and np.max(x) > 240
        if is_binary:
            xc = x
            xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
            xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
            one_pixel_edge_count = np.where(xc < x.squeeze())[0].shape[0]
            all_edge_count = np.where(x > 127)[0].shape[0]
            is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

    if 2 < unique_color_count < 200:
        interpolation = cv2.INTER_NEAREST
    elif new_size_is_smaller:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    # Ridimensiona l'immagine e la maschera, se presente
    y = cv2.resize(x, size, interpolation=interpolation)
    if inpaint_mask is not None:
        inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

    # Se l'immagine è binaria, applica ulteriori trasformazioni
    if is_binary:
        y = y.astype("uint8")
        # y_gray = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        _, y_bin = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y = cv2.cvtColor(y_bin, cv2.COLOR_GRAY2BGR)

    # Se c'è una maschera, aggiungila all'immagine ridimensionata
    if inpaint_mask is not None:
        inpaint_mask = (inpaint_mask > 127).astype(np.uint8) * 255
        y = np.concatenate([y, inpaint_mask[:, :, None]], axis=2)

    return y


safeint = lambda x: int(np.round(x))


def apply_border_noise(detected_map, outp, mask, h, w):
    # keep only the first 3 channels
    detected_map = detected_map[:, :, 0:3].copy()
    detected_map = detected_map.astype(np.float32)
    new_h, new_w, _ = mask.shape
    # calculate the ratio between the old and new image
    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w
    k = min(k0, k1)

    if outp == "outpainting":
        # find the borders of the mask
        border_pixels = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
        # calculate the median color for the borders
        high_quality_border_color = np.median(border_pixels, axis=0).astype(detected_map.dtype)
        # create the background with the same color
        high_quality_background = np.tile(high_quality_border_color[None, None], [safeint(h), safeint(w), 1])
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        img_rgba = np.zeros((high_quality_background.shape[0], high_quality_background.shape[1], 4), dtype=np.float32)
        img_rgba[:, :, 0:3] = high_quality_background
        img_rgba[:, :, 3] = 255  # create a 4 channel image with the alpha channel set to 1
        img_rgba_map = np.zeros((detected_map.shape[0], detected_map.shape[1], 4), dtype=np.float32)
        img_rgba_map[:, :, 0:3] = detected_map
        img_rgba_map[:, :, 3] = 0
        detected_map = img_rgba_map
        high_quality_background = img_rgba
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
    else:  # TODO: make sure that everything would work with inpaint
        # find the holes in the mask( where is equal to white)
        mask = mask.max(axis=2) > 254  # TODO: adapt this
        labeled, num_features = ndimage.label(mask)
        high_quality_background = np.zeros((safeint(old_h), safeint(old_w), 3))  # Make an empty image
        for i in range(1, num_features + 1):
            specific_mask = labeled == i

            # find the 'holes' borders
            borders = np.concatenate(
                [
                    detected_map[1:, :][specific_mask[1:, :] & ~specific_mask[:-1, :]],
                    detected_map[:-1, :][specific_mask[:-1, :] & ~specific_mask[1:, :]],
                    detected_map[:, 1:][specific_mask[:, 1:] & ~specific_mask[:, :-1]],
                    detected_map[:, :-1][specific_mask[:, :-1] & ~specific_mask[:, 1:]],
                ],
                axis=0,
            )

            # calculate the mean of the single borders
            high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)

            # fill the hole with its specific filling color
            high_quality_background[specific_mask] = high_quality_border_color
        assert high_quality_background.shape == detected_map.shape, "The images must have the same shape"

        # Create an empty rgba image
        result = np.zeros((high_quality_background.shape[0], high_quality_background.shape[1], 4), dtype=np.float32)

        # compy the bg in the empty image
        result[:, :, 0:3] = high_quality_background

        # set alpha channel to maks
        result[mask, 3] = 255.0
        result[~mask, 3] = 0.0

        # overlay detect map and high quality background
        alpha_channel = 1 - result[:, :, 3] / 255.0
        for i in range(3):  # RGB channels
            result[:, :, i] = (1 - alpha_channel) * result[:, :, i] + alpha_channel * detected_map[:, :, i]
        detected_map = result
    return get_pytorch_control(detected_map), detected_map


from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class VAEWrapper:
    def __init__(self, vae_instance):
        super().__init__()  # o qualsiasi altro argomento necessario per il costruttore di VAE
        self.vae = vae_instance
        self.scale_factor = 0.18215

    def to(self, where=None):
        if isinstance(where, torch.device):
            self.vae.first_stage_model.to(where)
        elif isinstance(where, torch.dtype):
            for module in self.vae.first_stage_model.modules():
                module.to(dtype=where)
        else:
            raise ValueError("Unsupported type for 'where' argument")

    def get_first_stage_encoding_(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage_(self, x):
        self.to(torch.device("cuda"))
        return self.vae.first_stage_model.encode(x)


class LaMaError(Exception):
    """An exception for the inpaint pipeline with LaMa preprocessor"""

    def __init__(self, message="You must provide a mask of the same size as the image or a horizontal/vertical expansion"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class abyz22_lamaPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "Width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 8}),
                "Width_Ratio_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Width_Ratio_max": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 8}),
                "Height_Ratio_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Height_Ratio_max": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    def _encode_image(self, vae, image):
        # encoder = VAEEncode()
        wrapper = VAEWrapper(vae)
        image = image[:, :, :, 0:3]
        image_without_alpha = image.to(wrapper.vae.vae_dtype) * 2.0 - 1.0
        image = image_without_alpha.permute(0, 3, 1, 2)
        # image = rearrange(image_without_alpha, "1 w h c -> 1 c w h")
        encoded_image = wrapper.encode_first_stage_(image)
        if torch.all(torch.isnan(encoded_image.mean())):
            print("All values produced are NANs, automatically upcasting dtype to float32")
            wrapper.to(torch.float32)
            encoded_image = wrapper.encode_first_stage_(image)
        vae_output = wrapper.get_first_stage_encoding_(encoded_image)
        return vae_output

    def preprocess(
        self, pixels: torch.Tensor, vae,  Width,Height,Width_Ratio_min,Width_Ratio_max,Height_Ratio_min,Height_Ratio_max, seed=0
    ):  # pixel= 1,768,512,3
        # 모드 설정
        print('☆★ '*20)
        np.random.seed(seed)
        random.seed(seed)

        model_lama = LamaInpainting()
        imgs, masks = None, None
        if Width_Ratio_min > Width_Ratio_max:
            Width_Ratio_min, Width_Ratio_max = Width_Ratio_max, Width_Ratio_min
        if Height_Ratio_min > Height_Ratio_max:
            Height_Ratio_min, Height_Ratio_max = Height_Ratio_max, Height_Ratio_min

        for i in range(pixels.shape[0]):
            ratio_h = round(np.random.uniform(Height_Ratio_min, Height_Ratio_max), 2)
            ratio_w = round(np.random.uniform(Width_Ratio_min, Width_Ratio_max), 2)
            Up, Left = int(round(Height * ratio_h)), int(round(Width * ratio_w))
            Down, Right = int(Height  - Up), int(Height - Left)

            copy_img = pixels.clone()[i, :, :, :]
            if Up + Down > 0:
                image = torch.rand((pixels.shape[1] + Up + Down, pixels.shape[2], pixels.shape[3]))
                up_start= np.random.uniform()
                image[Up : Up + pixels.shape[1], :, :] = copy_img

                # 마스크 제작하기
                mask = torch.ones_like(image[:, :, 0:1])
                mask[Up : Up + pixels.shape[1], :, :] = 0

                # 이미지 크기/값 lama에 맞추기 (0.0~1.0 -> 0~255)
                image_with_alpha = (torch.cat([image, mask], -1).numpy() * 255).astype(np.uint8)  # 알파채널 합침
                image_lama, remove_pad = resize_image_with_pad(image_with_alpha, 256, skip_hwc3=True)

                # lama에 이미지+마스크 넣기
                prd_color = model_lama(image_lama)
                # print("Up+Down :", image_lama.shape, prd_color.shape)
                # 이미지 크기 원상복구
                prd_color = remove_pad(prd_color)
                prd_color = cv2.resize(prd_color, (image.shape[1], image.shape[0]))

                raw_mask = image_with_alpha[:, :, 3:4]
                mask_alpha = raw_mask > 0
                # add alpha channel to the image
                final_img_with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
                final_img_with_alpha[:, :, 3] = np.where(mask_alpha.squeeze(), 255, 0)
                final_img_with_alpha[:, :, 0:3] = np.where(mask_alpha, prd_color, image_with_alpha[:, :, 0:3])
                copy_img = torch.from_numpy(final_img_with_alpha[:, :, 0:3]) / 255.0

            if Left + Right > 0:
                image = torch.rand((pixels.shape[1] + Up + Down, pixels.shape[2] + Left + Right, pixels.shape[3]))
                image[:, Left : Left + pixels.shape[2], :] = copy_img

                # 마스크 제작하기
                mask = torch.ones_like(image[:, :, 0:1])
                mask[:, Left : Left + pixels.shape[2], :] = 0

                # 이미지 크기/값 lama에 맞추기 (0.0~1.0 -> 0~255)
                image_with_alpha = (torch.cat([image, mask], -1).numpy() * 255).astype(np.uint8)  # 알파채널 합침
                # image_lama, remove_pad = resize_image_with_pad(image_with_alpha, 256, skip_hwc3=True)
                # image_lama= cv2.resize(image_with_alpha,(256,256))
                image_lama=image_with_alpha
                # lama에 이미지+마스크 넣기

                prd_color = model_lama(image_lama)
                # print("Left+Right", image_lama.shape, prd_color.shape)

                # 이미지 크기 원상복구
                # prd_color = remove_pad(prd_color)
                # prd_color = cv2.resize(prd_color, (image.shape[1], image.shape[0]))

                raw_mask = image_with_alpha[:, :, 3:4]
                mask_alpha = raw_mask > 0
                # add alpha channel to the image
                final_img_with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
                final_img_with_alpha[:, :, 3] = np.where(mask_alpha.squeeze(), 255, 0)
                final_img_with_alpha[:, :, 0:3] = np.where(mask_alpha, prd_color, image_with_alpha[:, :, 0:3])

            final_img_with_alpha = final_img_with_alpha[np.newaxis, :]
            # print(final_img_with_alpha.shape)

            mask = torch.ones_like(image[:, :, 0:1])
            mask[Up : Up + pixels.shape[1], Left : Left + pixels.shape[2], :] = 0
            mask = mask.permute(2, 0, 1).unsqueeze(0)
            if i == 0:
                imgs = final_img_with_alpha
                masks = mask
            else:
                imgs = np.concatenate([imgs, final_img_with_alpha])
                masks = torch.cat([masks, mask])

        image = (torch.from_numpy(imgs).float() / 255.0).clone().to("cuda" if torch.cuda.is_available() else "cpu")

        # 이미지 encode 넣고, masking 씌우기
        encoded_image = self._encode_image(vae, image)
        encoded_image_dict = {"samples": encoded_image.cpu()}
        mask_with_1 = torch.ones_like(image[:, :, :, 0])
        encoded_image_dict = SetLatentNoiseMask().set_mask(encoded_image_dict, mask_with_1)[0]

        pixels = pixels.permute(0, 3, 1, 2)
        img = torch.nn.functional.pad(pixels, (Left, Right, Up, Down), "constant", -1).permute(0, 2, 3, 1).to("cuda" if torch.cuda.is_available() else "cpu")
        return (img, encoded_image_dict)  # image= 1,h,w,c  0.0~1.0


    # masks.to("cuda" if torch.cuda.is_available() else "cpu")
    RETURN_TYPES = ("IMAGE", "LATENT", )
    RETURN_NAMES = ("LaMa Preprocessed Image", "LaMa Preprocessed Latent", )
    FUNCTION = "preprocess"
    CATEGORY = "abyz22"
