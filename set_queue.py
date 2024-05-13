import math, cv2, random, torch, torchvision, json, os
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드
from . import wildcards
from .utils import any_typ
from server import PromptServer
from . import utils

if os.path.isfile("./custom_nodes/image_control/prompts_maker.py"):
    from . import prompts_maker
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment

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


class abyz22_setimageinfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode_type": (["ab", "d"],),
                "Num of Prompts": ("INT", {"default": 1, "min": 0, "max": 30, "step": 1}),
                "image per Prompt": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("Mode_type", "Num of Prompts", "image per Prompt", "batch_size", "width", "height")

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(sefl, *args, **kwargs):
        return (
            kwargs["mode_type"],
            kwargs["Num of Prompts"],
            kwargs["image per Prompt"],
            kwargs["batch_size"],
            kwargs["width"],
            kwargs["height"],
        )


class abyz22_ImpactWildcardEncode_GetPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "Populate", "label_off": "Fixed"}),
                "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    CATEGORY = "abyz22"

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "clip", "conditioning", "populated_text")
    FUNCTION = "doit"

    @staticmethod
    def process_with_loras(**kwargs):
        return wildcards.process_with_loras(**kwargs)

    @staticmethod
    def get_wildcard_list():  # 이건 사용 안함 (왜있는지 모름)
        return wildcards.get_wildcard_list()

    def doit(self, *args, **kwargs):
        wildcard_process = nodes.NODE_CLASS_MAPPINGS["ImpactWildcardProcessor"].process

        populated = wildcard_process(text=kwargs["wildcard_text"], seed=kwargs["seed"])
        model, clip, conditioning = wildcards.process_with_loras(populated, kwargs["model"], kwargs["clip"])

        return (model, clip, conditioning, populated)


class abyz22_SetQueue:
    def __init__(self):
        self.cur_prompts, self.cur_batch = 0, 0
        self.highest_num = 0
        self.base_folder_path = ""
        self.prompt = ""
        self.debug = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode_type": ("STRING", {"default": "ab"}),
                "Num_of_Prompts": ("INT", {"default": 1, "min": 0, "max": 30, "step": 1}),
                "image_per_Prompt": ("INT", {"default": 24, "min": 1, "max": 30, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
                "prompt": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("Bool_trigger_queue", "folder_path", "prompt", "cur_prompts", "cur_batch", "debug")

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):
        cur_prompts, cur_batch = self.cur_prompts, self.cur_batch
        Bool_trigger_queue = False

        new_path = "no-data"

        if kwargs["Num_of_Prompts"] != 0:  # 폴더생성 1 이상이면
            if self.highest_num == 0:  # 최고폴더 번호 기본값=0 으로 되있으면, 최고폴더 찾기
                if kwargs["mode_type"] == "ab":
                    self.base_folder_path = "E:/ComfyUI/output/1submit/"
                elif kwargs["mode_type"] == "d":
                    self.base_folder_path = "E:/ComfyUI/output/3submit/"
                file_list = os.listdir(self.base_folder_path)
                file_list = [f for f in file_list if len(f.split(".")) == 1]
                file_list = [int(f.split(" ")[0]) for f in file_list]
                file_list.sort(reverse=True)
                self.highest_num = file_list[0] + 1

            if self.cur_batch == 0:  # 돌고돌아 현재 폴더 배치0되면(처음)
                if (kwargs["Num_of_Prompts"] == 1) and (kwargs["prompt"] != ""):  # 폴더1개만이고 프롬프트 있으면,
                    self.prompt = kwargs["prompt"]
                else:
                    self.prompt = prompts_maker.make_prompt(
                        mode=kwargs["mode_type"], etc=self.highest_num + 1 + self.cur_prompts, seed=kwargs["seed"]
                    )

                epath = "E:/ComfyUI/output/history.xlsx"
                T_workbook = load_workbook(epath, data_only=True)
                if kwargs["mode_type"] == "ab":
                    T_worksheet = T_workbook["Sheet1,2"]
                elif kwargs["mode_type"] == "d":
                    T_worksheet = T_workbook["Sheet4"]

                # find excel-index
                # new_idx = 0
                # for i in range(100, 9999):
                #     if T_worksheet[f"E{i}"].value == None:
                #         new_idx = i
                #         break
                # if new_idx == (self.highest_num + self.cur_prompts + 4):
                #     T_worksheet[f"E{new_idx}"].value = self.prompt
                # else:
                #     raise Exception("excel idx doesn't match")
                new_idx = self.highest_num + self.cur_prompts + 4

                T_worksheet[f"E{new_idx}"].value = self.prompt
                T_worksheet.row_dimensions[new_idx].height = 48
                T_worksheet[f"E{new_idx}"].alignment = Alignment(wrap_text=True)
                T_workbook.save(epath)

            new_path = self.base_folder_path + str(self.highest_num + self.cur_prompts)

        if kwargs["Num_of_Prompts"] == 0:
            Bool_trigger_queue = False
            cur_prompts, cur_batch = 0, 0
            self.highest_num, self.prompt = 0, ""
            self.debug = "Prompt testing...."

        elif cur_prompts < kwargs["Num_of_Prompts"]:  # 폴더 별
            if (cur_batch + kwargs["batch_size"]) < kwargs["image_per_Prompt"]:  # 폴더내 생성해야될 이미지 더 있으면
                Bool_trigger_queue = True
                cur_prompts, cur_batch = cur_prompts, cur_batch + kwargs["batch_size"]
                self.debug = f"{cur_prompts} / {cur_batch}"

            else:  # 폴더 내 이미지 생성 끝나면(폴더 내 마지막 틱)
                if (cur_prompts + 1) >= kwargs["Num_of_Prompts"]:  # 마지막 폴더일때
                    Bool_trigger_queue = False
                    cur_prompts, cur_batch = 0, 0
                    self.highest_num = 0
                    self.debug = "4444444"
                else:  # 생성해야할 폴더 남아있으면
                    Bool_trigger_queue = True
                    cur_prompts, cur_batch = cur_prompts + 1, 0
                    self.debug = "55555"

        self.cur_prompts, self.cur_batch = cur_prompts, cur_batch
        return Bool_trigger_queue, new_path, self.prompt, cur_prompts, cur_batch, self.debug


def workflow_to_map(workflow):
    nodes = {}
    links = {}
    for link in workflow["links"]:
        links[link[0]] = link[1:]
    for node in workflow["nodes"]:
        nodes[str(node["id"])] = node

    return nodes, links


class abyz22_bypass:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any_typ,),
                "mode": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Mute/Bypass"}),
                "behavior": ("BOOLEAN", {"default": True, "label_on": "Mute", "label_off": "Bypass"}),
                "prob": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "doit"

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    OUTPUT_NODE = True
    CATEGORY = "abyz22"

    @classmethod
    def IS_CHANGED(self, value, mode, behavior=True, unique_id=None, prompt=None, extra_pnginfo=None):
        nodes, links = workflow_to_map(extra_pnginfo["workflow"])

        next_nodes = []

        for link in nodes[unique_id]["outputs"][0]["links"]:
            node_id = str(links[link][2])
            utils.collect_non_reroute_nodes(nodes, links, next_nodes, node_id)

        return next_nodes

    def doit(self, value, mode, behavior=True, prob=0.5, unique_id=None, prompt=None, extra_pnginfo=None):
        global error_skip_flag

        nodes, links = workflow_to_map(extra_pnginfo["workflow"])

        active_nodes = []
        mute_nodes = []
        bypass_nodes = []

        for link in nodes[unique_id]["outputs"][0]["links"]:
            node_id = str(links[link][2])

            next_nodes = []
            utils.collect_non_reroute_nodes(nodes, links, next_nodes, node_id)

            for next_node_id in next_nodes:
                node_mode = nodes[next_node_id]["mode"]

                if node_mode == 0:
                    active_nodes.append(next_node_id)
                elif node_mode == 2:
                    mute_nodes.append(next_node_id)
                elif node_mode == 4:
                    bypass_nodes.append(next_node_id)

        if (mode == False) and (random.random() < prob):
            if behavior:  # mute
                should_be_mute_nodes = active_nodes + bypass_nodes
                if len(should_be_mute_nodes) > 0:
                    PromptServer.instance.send_sync(
                        "impact-bridge-continue", {"node_id": unique_id, "mutes": list(should_be_mute_nodes)}
                    )
                    error_skip_flag = True
                    raise Exception(
                        "IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE\nIf you see this message, your ComfyUI-Manager is outdated. Please update it."
                    )
            else:  # bypass
                should_be_bypass_nodes = active_nodes + mute_nodes
                if len(should_be_bypass_nodes) > 0:
                    PromptServer.instance.send_sync(
                        "impact-bridge-continue", {"node_id": unique_id, "bypasses": list(should_be_bypass_nodes)}
                    )
                    error_skip_flag = True
                    raise Exception(
                        "IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE\nIf you see this message, your ComfyUI-Manager is outdated. Please update it."
                    )
        else:
            # active
            should_be_active_nodes = mute_nodes + bypass_nodes
            if len(should_be_active_nodes) > 0:
                PromptServer.instance.send_sync(
                    "impact-bridge-continue", {"node_id": unique_id, "actives": list(should_be_active_nodes)}
                )
                error_skip_flag = True
                raise Exception(
                    "IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE\nIf you see this message, your ComfyUI-Manager is outdated. Please update it."
                )

        # if mode:
        #     # active
        #     should_be_active_nodes = mute_nodes + bypass_nodes
        #     if len(should_be_active_nodes) > 0:
        #         PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'actives': list(should_be_active_nodes)})
        #         error_skip_flag = True
        #         raise Exception("IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE\nIf you see this message, your ComfyUI-Manager is outdated. Please update it.")

        # elif behavior:
        #     # mute
        #     should_be_mute_nodes = active_nodes + bypass_nodes
        #     if len(should_be_mute_nodes) > 0:
        #         PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'mutes': list(should_be_mute_nodes)})
        #         error_skip_flag = True
        #         raise Exception("IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE\nIf you see this message, your ComfyUI-Manager is outdated. Please update it.")

        # else:
        #     # bypass
        #     should_be_bypass_nodes = active_nodes + mute_nodes
        #     if len(should_be_bypass_nodes) > 0:
        #         PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'bypasses': list(should_be_bypass_nodes)})
        #         error_skip_flag = True
        #         raise Exception("IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE\nIf you see this message, your ComfyUI-Manager is outdated. Please update it.")

        return (value,)
