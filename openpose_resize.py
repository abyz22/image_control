import math, cv2, random, torch, torchvision
import numpy as np
import nodes, folder_paths  # 기본노드, 파일로드

# 관절과 연결선의 색상 정의 BGR

# 관절과 연결선의 색상 정의 BGR
JOINT_COLORS = [
    [0, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [0, 255, 255],
    [0, 255, 170],
    [0, 255, 85],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [255, 255, 0],
    [255, 170, 0],
    [255, 85, 0],
    [255, 0, 0],
    [255, 0, 85],
    [255, 0, 170],
    [255, 0, 255],
    [170, 0, 255],
    [85, 0, 255],
]
LINE_COLORS_BODY = [
    [0, 0, 153],
    [0, 51, 153],
    [0, 102, 153],
    [0, 153, 153],
    [0, 153, 102],
    [0, 153, 51],
    [0, 153, 0],
    [51, 153, 0],
    [102, 153, 0],
    [153, 153, 0],
    [153, 102, 0],
    [153, 51, 0],
    [153, 0, 0],
    [153, 0, 51],
    [153, 0, 102],
    [153, 0, 153],
    [102, 0, 153],
]
LINE_COLORS_HAND = [
    [0, 0, 255],
    [0, 77, 255],
    [0, 102, 255],
    [0, 229, 225],
    [0, 255, 204],
    [0, 255, 128],  # 6
    [0, 255, 51],
    [25, 255, 0],
    [102, 255, 0],
    [179, 255, 0],
    [255, 255, 0],  # 11
    [255, 178, 0],
    [255, 102, 0],
    [255, 25, 0],
    [255, 0, 51],
    [255, 0, 128],  # 16
    [255, 0, 204],
    [230, 0, 255],
    [153, 0, 255],
    [77, 0, 255],
]
# 코코 모델의 18개 관절 연결 정보
POSE_PAIRS_BODY = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]
POSE_PAIRS_HAND = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


# 포즈 그리기 함수
def draw_pose(image, keypoints, pose_pairs, joint_colors, line_colors, thickness=2):
    if keypoints.shape[0] == 18:  # 18,70,21
        for pair, line_color in zip(pose_pairs, line_colors):
            partA = pair[0]
            partB = pair[1]

            if keypoints[partA][2] > 0.1 and keypoints[partB][2] > 0.1:
                cv2.line(
                    image,
                    (int(keypoints[partA][0]), int(keypoints[partA][1])),
                    (int(keypoints[partB][0]), int(keypoints[partB][1])),
                    line_color,
                    thickness,
                )
        for (kp_x, kp_y, conf), j_color in zip(keypoints, joint_colors):
            if (kp_x > 0.0 or kp_y > 0.0) and (conf > 0.1):  # 값이 0.0보다 하나씩이라도 높고& conf값 0.1 넘으면
                cv2.circle(image, (int(kp_x), int(kp_y)), 5, j_color, -1)
        return image

    elif keypoints.shape[0] == 70:
        for kp_x, kp_y, conf in keypoints:
            if (kp_x > 0.0 or kp_y > 0.0) and (conf > 0.1):  # 값이 0.0보다 하나씩이라도 높고& conf값 0.1 넘으면
                cv2.circle(image, (int(kp_x), int(kp_y)), 3, [255, 255, 255], -1)
        return image

    elif keypoints.shape[0] == 21:
        for pair, line_color in zip(pose_pairs, line_colors):
            partA = pair[0]
            partB = pair[1]

            if keypoints[partA][2] > 0.1 and keypoints[partB][2] > 0.1:
                cv2.line(
                    image,
                    (int(keypoints[partA][0]), int(keypoints[partA][1])),
                    (int(keypoints[partB][0]), int(keypoints[partB][1])),
                    line_color,
                    thickness,
                )
        for kp_x, kp_y, conf in keypoints:
            if (kp_x > 0.0 or kp_y > 0.0) and (conf > 0.1):  # 값이 0.0보다 하나씩이라도 높고& conf값 0.1 넘으면
                cv2.circle(image, (int(kp_x), int(kp_y)), 3, [255, 0, 0], -1)
        return image


# 파라미터에 따른 길이 조정 함수
def adjust_length(pointA, pointB, scale):
    xA, yA = pointA[:2]
    xB, yB = pointB[:2]
    vector = np.array([int(xB - xA), int(yB - yA)])
    length = np.linalg.norm(vector)
    if length == 0:
        return pointB[:2]
    unit_vector = vector / length
    new_point = np.array(pointA[:2]) + unit_vector * (length * (scale - 1))
    return int(new_point[0]), int(new_point[1])


def rescale_pt(a, b, keypoints, parameters, param, ptmove_list, double=1):
    new_b = adjust_length(keypoints[a], keypoints[b], parameters[param])
    dx, dy = (new_b - keypoints[a][:2]) / double

    for ptmove in ptmove_list:
        keypoints[ptmove][:2] = keypoints[ptmove][:2] + [int(dx), int(dy)]
    return keypoints


# 새로운 키포인트 데이터를 생성하는 함수
def rescale_keypoints(keypoints, parameters):
    resize_param = [
        "Head Size",
        "Neck",
        "Shoulder Width",
        "Shoulder To Hip",
        "Fore Arm",
        "Upper Arm",
        "Hips",
        "Thigh",
        "Lower Leg",
        "Face",
        "Hands",
    ]
    for param in resize_param:
        if (parameters[param] >= 1.00001 or parameters[param] <= 0.99999) and keypoints.shape[0] == 18:
            if False:
                pass
            elif param == "Neck":
                ptmove_list = [0, 14, 15, 16, 17]
                keypoints = rescale_pt(1, 0, keypoints, parameters, param, ptmove_list)

            elif param == "Shoulder Width":
                ptmove_list = [2, 3, 4]
                keypoints = rescale_pt(1, 2, keypoints, parameters, param, ptmove_list)
                ptmove_list = [5, 6, 7]
                keypoints = rescale_pt(1, 5, keypoints, parameters, param, ptmove_list)
            elif param == "Shoulder To Hip":
                ptmove_list = [8, 9, 10]
                keypoints = rescale_pt(1, 8, keypoints, parameters, param, ptmove_list)
                ptmove_list = [11, 12, 13]
                keypoints = rescale_pt(1, 11, keypoints, parameters, param, ptmove_list)
            elif param == "Fore Arm":
                ptmove_list = [4]
                keypoints = rescale_pt(3, 4, keypoints, parameters, param, ptmove_list)
                ptmove_list = [7]
                keypoints = rescale_pt(6, 7, keypoints, parameters, param, ptmove_list)
            elif param == "Upper Arm":
                ptmove_list = [3, 4]
                keypoints = rescale_pt(2, 3, keypoints, parameters, param, ptmove_list)
                ptmove_list = [6, 7]
                keypoints = rescale_pt(5, 6, keypoints, parameters, param, ptmove_list)
            elif param == "Hips":
                ptmove_list = [8, 9, 10]
                keypoints = rescale_pt(11, 8, keypoints, parameters, param, ptmove_list, double=2)  # 처음 50% 늘리고
                ptmove_list = [11, 12, 13]
                keypoints = rescale_pt(8, 11, keypoints, parameters, param, ptmove_list, double=3)  # 1.5에서 30% 더하면 2나옴
            elif param == "Thigh":
                ptmove_list = [9, 10]
                keypoints = rescale_pt(8, 9, keypoints, parameters, param, ptmove_list)
                ptmove_list = [12, 13]
                keypoints = rescale_pt(11, 12, keypoints, parameters, param, ptmove_list)
            elif param == "Lower Leg":
                ptmove_list = [10]
                keypoints = rescale_pt(9, 10, keypoints, parameters, param, ptmove_list)
                ptmove_list = [13]
                keypoints = rescale_pt(12, 13, keypoints, parameters, param, ptmove_list)
            elif param == "Head Size":
                ptmove_list = [14, 16]
                keypoints = rescale_pt(0, 14, keypoints, parameters, param, ptmove_list)
                ptmove_list = [16]
                keypoints = rescale_pt(14, 16, keypoints, parameters, param, ptmove_list)
                ptmove_list = [15, 17]
                keypoints = rescale_pt(0, 15, keypoints, parameters, param, ptmove_list)
                ptmove_list = [17]
                keypoints = rescale_pt(15, 17, keypoints, parameters, param, ptmove_list)
            elif param == "Head Size":
                ptmove_list = [14, 16]
                keypoints = rescale_pt(0, 14, keypoints, parameters, param, ptmove_list)
                ptmove_list = [16]
                keypoints = rescale_pt(14, 16, keypoints, parameters, param, ptmove_list)
                ptmove_list = [15, 17]
                keypoints = rescale_pt(0, 15, keypoints, parameters, param, ptmove_list)
                ptmove_list = [17]
                keypoints = rescale_pt(15, 17, keypoints, parameters, param, ptmove_list)

        if (parameters[param] >= 1.00001 or parameters[param] <= 0.99999) and keypoints.shape[0] == 70:
            if param == "Face":
                ptmove_list = list(range(70))
                ptmove_list.remove(30)
                for p in ptmove_list:
                    keypoints = rescale_pt(30, p, keypoints, parameters, param, [p])

        if (parameters[param] >= 1.00001 or parameters[param] <= 0.99999) and keypoints.shape[0] == 21:
            if param == "Hands":
                ptmove_list = list(range(21))
                ptmove_list.remove(0)
                for p in ptmove_list:
                    keypoints = rescale_pt(0, p, keypoints, parameters, param, [p])

    return keypoints


class abyz22_ResizeOpenpose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # image= N,768,512,3
                "POSE_KEYPOINT": ("POSE_KEYPOINT",),
                "Head Size": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Neck": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Shoulder Width": (
                    "FLOAT",
                    {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"},
                ),
                "Shoulder To Hip": (
                    "FLOAT",
                    {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"},
                ),
                "Fore Arm": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Upper Arm": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Hips": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Thigh": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Lower Leg": ("FLOAT", {"default": 1, "min": 0.33, "max": 3, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Face": ("FLOAT", {"default": 1, "min": 0.0, "max": 2, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Hands": ("FLOAT", {"default": 1, "min": 0.0, "max": 2, "step": 0.01, "round": 0.001, "dispaly": "slider"}),
                "Line Thickness": (
                    "INT",
                    {"default": 5, "min": 1, "max": 10, "step": 1, "round": 0.1, "dispaly": "slider"},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "abyz22"

    def run(self, *args, **kwargs):
        image = kwargs["image"]

        image_width, image_height = kwargs["image"].shape[2], kwargs["image"].shape[1]

        # 데이터에서 사람의 포즈 정보 추출 및 관절 길이 조정
        images = None
        for i, _ in enumerate(range(kwargs["image"].shape[0])):
            image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            for person in kwargs["POSE_KEYPOINT"][i]["people"]:
                keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
                nose, right_hand, left_hand = keypoints[0].copy(), keypoints[4].copy(), keypoints[7].copy()
                keypoints = rescale_keypoints(keypoints, kwargs)
                image = draw_pose(image, keypoints, POSE_PAIRS_BODY, JOINT_COLORS, LINE_COLORS_BODY, kwargs["Line Thickness"])

                if kwargs["Face"] != 0:
                    face_keypoints = np.array(person["face_keypoints_2d"]).reshape(-1, 3)
                    face_keypoints = face_keypoints + (keypoints[0] - nose)
                    face_keypoints = rescale_keypoints(face_keypoints, kwargs)
                    image = draw_pose(
                        image, face_keypoints, POSE_PAIRS_BODY, JOINT_COLORS, LINE_COLORS_BODY, kwargs["Line Thickness"]
                    )

                if kwargs["Hands"] != 0:
                    hand_right_keypoints = np.array(person["hand_right_keypoints_2d"]).reshape(-1, 3)
                    hand_right_keypoints = hand_right_keypoints + (keypoints[4] - right_hand)
                    hand_right_keypoints = rescale_keypoints(hand_right_keypoints, kwargs)
                    image = draw_pose(
                        image, hand_right_keypoints, POSE_PAIRS_HAND, JOINT_COLORS, LINE_COLORS_HAND, kwargs["Line Thickness"]
                    )

                    hand_left_keypoints = np.array(person["hand_left_keypoints_2d"]).reshape(-1, 3)
                    hand_left_keypoints = hand_left_keypoints + (keypoints[7] - left_hand)
                    hand_left_keypoints = rescale_keypoints(hand_left_keypoints, kwargs)
                    image = draw_pose(
                        image, hand_left_keypoints, POSE_PAIRS_HAND, JOINT_COLORS, LINE_COLORS_HAND, kwargs["Line Thickness"]
                    )

            image = image[:, :, ::-1][np.newaxis, :, :, :]
            images = image if i == 0 else np.concatenate([images, image])

        return (torch.Tensor(images / 255.0),)
