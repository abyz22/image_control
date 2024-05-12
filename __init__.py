from .outpainting_node import *
from .wildcardencoder import *
from .save_image import *
from .set_queue import *
from .lowrise import *
from .non_switch import *
from .small_head import *
from .outpainting_lama import *
from .utils import *
from .random_mask import *
from .censoring import *

NODE_CLASS_MAPPINGS = {
    "abyz22_Padding Image": abyz22_Pad_Image,
    "abyz22_ImpactWildcardEncode": abyz22_ImpactWildcardEncode,
    # "abyz22_Ksampler": abyz22_KSampler,
    # "abyz22_ToBasicPipe": abyz22_ToBasicPipe,
    # "abyz22_FromBasicPipe_v2": abyz22_FromBasicPipe_v2,
    "abyz22_setimageinfo": abyz22_setimageinfo,
    "abyz22_SaveImage": abyz22_SaveImage,
    "abyz22_ImpactWildcardEncode_GetPrompt": abyz22_ImpactWildcardEncode_GetPrompt,
    "abyz22_SetQueue": abyz22_SetQueue,
    "abyz22_drawmask": abyz22_drawmask,
    "abyz22_FirstNonNull": abyz22_FirstNonNull,
    "abyz22_blendimages": abyz22_blendimages,
    "abyz22_blend_onecolor": abyz22_blend_onecolor,
    "abyz22_bypass": abyz22_bypass,
    "abyz22_smallhead": abyz22_smallhead,
    "abyz22_lamaPreprocessor": abyz22_lamaPreprocessor,
    "abyz22_lamaInpaint": abyz22_lamaInpaint,
    "abyz22_makecircles": abyz22_makecircles,
    "abyz22_Topipe": abyz22_Topipe,
    "abyz22_Frompipe": abyz22_Frompipe,
    "abyz22_Editpipe": abyz22_Editpipe,
    "abyz22_Convertpipe": abyz22_Convertpipe,
    "abyz22_RemoveControlnet": abyz22_RemoveControlnet,
    "abyz22_RandomMask": abyz22_RandomMask,
    "abyz22_AddPrompt": abyz22_AddPrompt,
    "abyz22_censoring": abyz22_censoring,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "abyz22_Padding Image": "abyz22_Padding Image",
    "abyz22_ImpactWildcardEncode": "abyz22_ImpactWildcardEncode",
    # "abyz22_Ksampler": "abyz22_Ksampler",
    # "abyz22_ToBasicPipe": "abyz22_ToBasicPipe",
    # "abyz22_FromBasicPipe_v2": "abyz22_FromBasicPipe_v2",
    "abyz22_setimageinfo": "abyz22_setimageinfo",
    "abyz22_SaveImage": "abyz22_SaveImage",
    "abyz22_ImpactWildcardEncode_GetPrompt": "abyz22_ImpactWildcardEncode_GetPrompt",
    "abyz22_SetQueue": "abyz22_SetQueue",
    "abyz22_drawmask": "abyz22_drawmask",
    "abyz22_FirstNonNull": "abyz22_FirstNonNull",
    "abyz22_blendimages": "abyz22_blendimages",
    "abyz22_blend_onecolor": "abyz22_blend_onecolor",
    "abyz22_bypass": "abyz22_bypass",
    "abyz22_smallhead": "abyz22_smallhead",
    "abyz22_lamaPreprocessor": "abyz22_lamaPreprocessor",
    "abyz22_lamaInpaint": "abyz22_lamaInpaint",
    "abyz22_makecircles": "abyz22_makecircles",
    "abyz22_Topipe": "abyz22_Topipe",
    "abyz22_Frompipe": "abyz22_Frompipe",
    "abyz22_Editpipe": "abyz22_Editpipe",
    "abyz22_Convertpipe": "abyz22_Convertpipe",
    "abyz22_RemoveControlnet": "abyz22_RemoveControlnet",
    "abyz22_RandomMask": "abyz22_RandomMask",
    "abyz22_AddPrompt": "abyz22_AddPrompt",
    "abyz22_censoring": "abyz22_censoring",
}
