import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


class abyz22_FirstNonNull:
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("VALUE",)
    FUNCTION = "FirstNonNull"
    CATEGORY = "abyz22"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {},
            "optional":{
                "value1": (ANY_TYPE,),
                "value2": (ANY_TYPE,),
            }
        }

    def FirstNonNull(self, value1=None, value2=None) -> ANY_TYPE:
        if value1 is None and value2 is None:
            raise Exception("Atleast One Value Input Expected")
        VALUE = value1 if value1 is not None else value2
        return (VALUE,)


