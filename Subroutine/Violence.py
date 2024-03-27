from dis import dis
from os import stat
import numpy as np


class violence:

    def __init__(self):
        # 暴力的種類
        self.types = {
            # person1(右手) 與 person2 最近距離
            "p1_right_hand_to_p2": {"abuser": "person1", "part": "r_wrist", "victim": "person2", "limit": -1, },
            # person1(左手) 與 person2 最近距離
            "p1_left_hand_to_p2": {"abuser": "person1", "part": "l_wrist", "victim": "person2", "limit": -1, },
            # person2(右手) 與 person1 最近距離
            "p2_right_hand_to_p1": {"abuser": "person2", "part": "r_wrist", "victim": "person1", "limit": -1, },
            # person2(左手) 與 person1 最近距離
            "p2_left_hand_to_p1": {"abuser": "person2", "part": "l_wrist", "victim": "person1", "limit": -1, },
            # person1(右腳) 與 person2 最近距離
            "p1_right_foot_to_p2": {"abuser": "person1", "part": "r_ankle", "victim": "person2", "limit": -1, },
            # person1(左腳) 與 person2 最近距離
            "p1_left_foot_to_p2": {"abuser": "person1", "part": "l_ankle", "victim": "person2", "limit": -1, },
            # person2(右腳) 與 person1 最近距離
            "p2_right_foot_to_p1": {"abuser": "person2", "part": "r_ankle", "victim": "person1", "limit": -1, },
            # person2(左腳) 與 person1 最近距離
            "p2_left_foot_to_p1": {"abuser": "person2", "part": "l_ankle", "victim": "person1", "limit": -1, },
        }

        self.threshold = {
            "center_acceleration": 13.57,
            "wrist_acceleration": 21.00,
            "ankle_acceleration": 10.52
        }
    pass

    def convert_poses_2d_to_features(self, poses_2d):  # 轉換資料格式
        features = {}
        for pose_id in range(len(poses_2d)):  # 幾個人

            pose = np.array(poses_2d[pose_id][0:-1]
                            ).reshape((-1, 3)).transpose()
            neck = tuple(pose[0:2, 0].astype(int))
            nose = tuple(pose[0:2, 1].astype(int))
            l_eye = tuple(pose[0:2, 16].astype(int))
            l_ear = tuple(pose[0:2, 18].astype(int))
            l_shoulder = tuple(pose[0:2, 3].astype(int))
            l_elbow = tuple(pose[0:2, 4].astype(int))
            l_wrist = tuple(pose[0:2, 5].astype(int))
            l_hip = tuple(pose[0:2, 6].astype(int))
            l_knee = tuple(pose[0:2, 7].astype(int))
            l_ankle = tuple(pose[0:2, 8].astype(int))
            r_eye = tuple(pose[0:2, 15].astype(int))
            r_ear = tuple(pose[0:2, 17].astype(int))
            r_shoulder = tuple(pose[0:2, 9].astype(int))
            r_elbow = tuple(pose[0:2, 10].astype(int))
            r_wrist = tuple(pose[0:2, 11].astype(int))
            r_hip = tuple(pose[0:2, 12].astype(int))
            r_knee = tuple(pose[0:2, 13].astype(int))
            r_ankle = tuple(pose[0:2, 14].astype(int))

            feature = {'neck': neck,
                       'nose': nose,
                       'l_eye': l_eye,
                       'l_ear': l_ear,
                       'l_shoulder': l_shoulder,
                       'l_elbow': l_elbow,
                       'l_wrist': l_wrist,
                       'l_hip': l_hip,
                       'l_knee': l_knee,
                       'l_ankle': l_ankle,
                       'r_eye': r_eye,
                       'r_ear': r_ear,
                       'r_shoulder': r_shoulder,
                       'r_elbow': r_elbow,
                       'r_wrist': r_wrist,
                       'r_hip': r_hip,
                       'r_knee': r_knee,
                       'r_ankle': r_ankle
                       }

            features["person" + str(pose_id + 1)] = feature
        pass

        return features
    pass

    def show_value(self, pose_id, pose):  # 顯示每個節點
        '''
        body_edges = np.array(
        [[0, 1],  # neck - nose
        [1, 16], [16, 18],  # nose - l_eye - l_ear
        [1, 15], [15, 17],  # nose - r_eye - r_ear
        [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
        [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
        [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
        [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle
        '''

        neck = tuple(pose[0:2, 0].astype(int))
        nose = tuple(pose[0:2, 1].astype(int))
        l_eye = tuple(pose[0:2, 16].astype(int))
        l_ear = tuple(pose[0:2, 18].astype(int))
        l_shoulder = tuple(pose[0:2, 3].astype(int))
        l_elbow = tuple(pose[0:2, 4].astype(int))
        l_wrist = tuple(pose[0:2, 5].astype(int))
        l_hip = tuple(pose[0:2, 6].astype(int))
        l_knee = tuple(pose[0:2, 7].astype(int))
        l_ankle = tuple(pose[0:2, 8].astype(int))
        r_eye = tuple(pose[0:2, 15].astype(int))
        r_ear = tuple(pose[0:2, 17].astype(int))
        r_shoulder = tuple(pose[0:2, 9].astype(int))
        r_elbow = tuple(pose[0:2, 10].astype(int))
        r_wrist = tuple(pose[0:2, 11].astype(int))
        r_hip = tuple(pose[0:2, 12].astype(int))
        r_knee = tuple(pose[0:2, 13].astype(int))
        r_ankle = tuple(pose[0:2, 14].astype(int))

        print("第 {} 位".format(pose_id + 1))
        print("左　耳:{}".format(l_ear), end="  ")
        print("左　眼:{}".format(l_eye), end="  ||  ")
        print("鼻　子:{}".format(nose), end="")
        print("右　眼:{}".format(r_eye), end="  ||  ")
        print("右　耳:{}".format(r_ear), end="\n")

        print("脖 子:{}".format(neck), end="\n")

        print("左肩膀:{}".format(l_shoulder), end="  ")
        print("左手肘:{}".format(l_elbow), end="  ")
        print("左手腕:{}".format(l_wrist), end="  ||  ")
        print("右肩膀:{}".format(r_shoulder), end="  ")
        print("右手肘:{}".format(r_elbow), end="  ")
        print("右手腕:{}".format(r_wrist), end="\n")

        print("左髖骨:{}".format(l_hip), end="  ")
        print("左膝蓋:{}".format(l_knee), end="  ")
        print("左腳踝:{}".format(l_ankle), end="  ||  ")
        print("右髖骨:{}".format(r_hip), end="  ")
        print("右膝蓋:{}".format(r_knee), end="  ")
        print("右腳踝:{}".format(r_ankle), end="\n")
    pass

    def identify_person(self, previous_features, current_features):  # 人員校正
        correction_features = {}
        person_move_distance = []
        temp_distance = 0

        # print(previous_features)
        # print()
        # print(current_features)
        for previous_values in previous_features.values():
            # print(previous_values["r_wrist"], previous_values["l_wrist"])
            for current_values in current_features.values():
                temp_distance += self.distance(
                    previous_values["neck"], current_values["neck"])
                person_move_distance.append(temp_distance)
                temp_distance = 0
            pass
        pass
        # print(person_move_distance)
        if len(person_move_distance) > 0:
            for index in range(0, len(person_move_distance), len(current_features)):
                scope = person_move_distance[index:index +
                                             len(current_features)]
                # print(scope)
                min_index = scope.index(min(scope))
                correction_features["person" + str(index // len(current_features) + 1)] = current_features[list(
                    current_features)[min_index % len(current_features)]]
            pass
        pass

        person_move_distance.clear()
        # print("================")

        return correction_features
    pass

    def calculate_closest_parts_distance(self, features):
        closest_parts_distance = {}

        for type in self.types:
            abuser = features[self.types[type]["abuser"]]  # 打人的
            part = abuser[self.types[type]["part"]]  # 打人的部位（手,腳）

            victim = features[self.types[type]["victim"]]  # 被打的
            closest_parts_distance[type] = self.calculate_part_to_person_distsnce_min(
                part, victim)  # 計算那一種型態特定部位（手,腳）距離另一個人的最近距離
        pass

        return closest_parts_distance
    pass

    def calculate_part_to_person_distsnce_min(self, part, person):
        if part == (-1, -1):  # 代表 "此部位" 未被偵測到
            return -1
        else:
            part_to_person_distsnces_min = {}  # 指定部位（手,腳）距離另一個人的最近節點距離
            for key, value in person.items():  # 人的所有節點
                if value != (-1, -1):
                    # 部位跟每個節點的距離
                    part_distance = self.distance(part, value)
                    if len(part_to_person_distsnces_min) == 0:
                        part_to_person_distsnces_min[key] = part_distance
                    else:
                        if part_to_person_distsnces_min[list(part_to_person_distsnces_min)[0]] > part_distance:
                            part_to_person_distsnces_min.clear()
                            part_to_person_distsnces_min[key] = part_distance
                        pass
                    pass
                pass
            pass
            return part_to_person_distsnces_min
        pass
    pass

    def distance(self, x, y):  # 計算兩點距離
        np_x = np.array(x)  # 人的部位
        np_y = np.array(y)  # 人的所有節點

        # 手跟每個節點的距離
        hand_distance = np.linalg.norm(np_x - np_y)
        return hand_distance
    pass

    def calculate_touch_limit(self, features):  # 計算碰觸上距離
        person1 = features["person1"]
        person2 = features["person2"]

        touch_distance_limit = {"person1": -1, "person2": -1}

        scale = 2 / 3

        # ========= person1 觸碰上限距離 ========= #
        distance_left = -1
        distance_right = -1

        if person1["l_hip"] != (-1, -1):
            distance_left = self.distance(person1["neck"], person1["l_hip"])
        pass

        if person1["r_hip"] != (-1, -1):
            distance_right = self.distance(person1["neck"], person1["r_hip"])
        pass

        touch_distance_limit["person1"] = max(
            distance_left, distance_right) * scale

        # ========= person2 觸碰上限距離 ========= #
        distance_left = -1
        distance_right = -1

        if person2["l_hip"] != (-1, -1):
            distance_left = self.distance(person2["neck"], person2["l_hip"])
        pass

        if person2["r_hip"] != (-1, -1):
            distance_right = self.distance(person2["neck"], person2["r_hip"])
        pass

        touch_distance_limit["person2"] = max(
            distance_left, distance_right) * scale

        return touch_distance_limit
    pass

    def rule_base_model(self, history_features, closest_parts_distance):
        # 最新一筆的資料
        current_features = history_features[list(history_features)[-1]]

        # 碰觸得距離上限
        touch_distance_limit = self.calculate_touch_limit(current_features)

        # 打人的那個人所能碰觸的距離上限
        for key, value in self.types.items():
            self.types[key]["limit"] = touch_distance_limit[self.types[key]["abuser"]]
        pass

        values = {}

        for key, value in closest_parts_distance.items():
            if value == -1:
                values[key] = -1
            else:
                values[key] = {
                    "center_distance": 0,
                    "center_acceleration": 0,
                    "wrist_distance": 0,
                    "wrist_acceleration": 0,
                    "ankle_distance": 0,
                    "ankle_acceleration": 0,
                    "status": 0
                }

                limit = self.types[key]["limit"]
                distance = value[list(value)[0]]

                # 最舊的資料
                oldest_features = history_features[list(
                    history_features)[0]]

                # 最新的資料
                current_features = history_features[list(
                    history_features)[-1]]

                parts_acceleration = {}

                for (_, oldest_value), (person_key, current_value) in zip(oldest_features.items(), current_features.items()):
                    parts_acceleration[person_key] = {
                        "r_wrist": 0,
                        "l_wrist": 0,
                        "r_ankle": 0,
                        "l_ankle": 0,
                        "neck": 0
                    }

                    for part in parts_acceleration[person_key].keys():
                        if oldest_value[part] == (-1, -1) or current_value[part] == (-1, -1):
                            parts_acceleration[person_key][part] = -1
                        else:
                            parts_acceleration[person_key][part] += self.distance(
                                oldest_value[part], current_value[part])
                        pass
                    pass
                pass

                oldest_distance_between = self.distance(
                    oldest_features["person1"]["neck"], oldest_features["person2"]["neck"])
                current_distance_between = self.distance(
                    current_features["person1"]["neck"], current_features["person2"]["neck"])

                abuser = self.types[key]["abuser"]
                part = self.types[key]["part"]

                acceleration = parts_acceleration[abuser][part]
                if acceleration == -1:
                    values[key] = -1
                else:
                    # 兩人中心之間的距離,如果可以碰到數值就會是正的
                    values[key]["center_distance"] = (
                        limit * 5) - current_distance_between

                    # 兩人中心之間的距離變化
                    values[key]["center_acceleration"] = current_distance_between - \
                        oldest_distance_between

                    if part[2:] == "wrist":
                        # 手跟另一個人之間的距離,如果有打到數值就會是正的
                        values[key]["wrist_distance"] = limit - distance

                        # 手的加速度
                        values[key]["wrist_acceleration"] = acceleration

                    elif part[2:] == "ankle":
                        # 腳跟另一個人之間的距離,如果有踢到數值就會是正的
                        values[key]["ankle_distance"] = limit - distance

                        # 腳的加速度
                        values[key]["ankle_acceleration"] = acceleration
                    pass

                    status = 0
                    if values[key]["center_distance"] > 0:  # 兩人之間的距離可以碰到
                        # print(values[key]["center_distance"])
                        # 手或腳可以碰到
                        if values[key]["wrist_distance"] > 0 or values[key]["ankle_distance"] > 0:
                            if values[key]["center_acceleration"] > self.threshold["center_acceleration"]:
                                status = 1  # violence
                            elif part[2:] == "wrist" and values[key]["wrist_acceleration"] > self.threshold["wrist_acceleration"]:
                                status = 1  # violence
                            elif part[2:] == "ankle" and values[key]["ankle_acceleration"] > self.threshold["ankle_acceleration"]:
                                status = 1  # violence
                            pass
                        pass
                    pass

                    values[key]["status"] = status

                pass
            pass
        pass
        return values
    pass

    def calculate_features(self, history_features, closest_parts_distance):
        # 最新一筆的資料
        current_features = history_features[list(history_features)[-1]]

        # 碰觸得距離上限
        touch_distance_limit = self.calculate_touch_limit(current_features)

        # 打人的那個人所能碰觸的距離上限
        for key, value in self.types.items():
            self.types[key]["limit"] = touch_distance_limit[self.types[key]["abuser"]]
        pass

        values = {}
        for key, value in closest_parts_distance.items():
            if value == -1:
                values[key] = -1
            else:
                values[key] = {
                    "center_distance": 0,
                    "center_acceleration": 0,
                    "wrist_distance": 0,
                    "wrist_acceleration": 0,
                    "ankle_distance": 0,
                    "ankle_acceleration": 0,
                }

                limit = self.types[key]["limit"]
                distance = value[list(value)[0]]

                # 最舊的資料
                oldest_features = history_features[list(
                    history_features)[0]]

                # 最新的資料
                current_features = history_features[list(
                    history_features)[-1]]

                parts_acceleration = {}

                for (_, oldest_value), (person_key, current_value) in zip(oldest_features.items(), current_features.items()):
                    parts_acceleration[person_key] = {
                        "r_wrist": 0,
                        "l_wrist": 0,
                        "r_ankle": 0,
                        "l_ankle": 0,
                        "neck": 0
                    }

                    for part in parts_acceleration[person_key].keys():
                        if oldest_value[part] == (-1, -1) or current_value[part] == (-1, -1):
                            parts_acceleration[person_key][part] = -1
                        else:
                            parts_acceleration[person_key][part] += self.distance(
                                oldest_value[part], current_value[part])
                        pass
                    pass
                pass

                oldest_distance_between = self.distance(
                    oldest_features["person1"]["neck"], oldest_features["person2"]["neck"])
                current_distance_between = self.distance(
                    current_features["person1"]["neck"], current_features["person2"]["neck"])

                abuser = self.types[key]["abuser"]
                part = self.types[key]["part"]

                acceleration = parts_acceleration[abuser][part]
                if acceleration == -1:
                    values[key] = -1
                else:
                    # 兩人中心之間的距離
                    values[key]["center_distance"] = (
                        limit * 2) - current_distance_between

                    # 兩人中心之間的距離變化
                    values[key]["center_acceleration"] = current_distance_between - \
                        oldest_distance_between

                    if part[2:] == "wrist":
                        # 手跟另一個人之間的距離,如果有打到數值就會是正的
                        values[key]["wrist_distance"] = limit - distance

                        # 手的加速度
                        values[key]["wrist_acceleration"] = acceleration

                    elif part[2:] == "ankle":
                        # 腳跟另一個人之間的距離,如果有踢到數值就會是正的
                        values[key]["ankle_distance"] = limit - distance

                        # 腳的加速度
                        values[key]["ankle_acceleration"] = acceleration
                    pass
                pass
            pass
        pass
        return values
    pass

    # 畫盒鬚圖用
    def calculate_features_original_value(self, history_features, closest_parts_distance):
        # 最新一筆的資料
        current_features = history_features[list(history_features)[-1]]

        # 碰觸得距離上限
        touch_distance_limit = self.calculate_touch_limit(current_features)

        # 打人的那個人所能碰觸的距離上限
        for key, value in self.types.items():
            self.types[key]["limit"] = touch_distance_limit[self.types[key]["abuser"]]
        pass

        values = {}

        for key, value in closest_parts_distance.items():
            if value == -1:
                values[key] = -1
            else:
                values[key] = {
                    "center_distance": 0,
                    "center_acceleration": 0,
                    "wrist_distance": 0,
                    "wrist_acceleration": 0,
                    "ankle_distance": 0,
                    "ankle_acceleration": 0,
                    "status": 0
                }
                distance = value[list(value)[0]]

                # 最舊的資料
                oldest_features = history_features[list(
                    history_features)[0]]

                # 最新的資料
                current_features = history_features[list(
                    history_features)[-1]]

                parts_acceleration = {}

                for (_, oldest_value), (person_key, current_value) in zip(oldest_features.items(), current_features.items()):
                    parts_acceleration[person_key] = {
                        "r_wrist": 0,
                        "l_wrist": 0,
                        "r_ankle": 0,
                        "l_ankle": 0,
                        "neck": 0
                    }

                    for part in parts_acceleration[person_key].keys():
                        if oldest_value[part] == (-1, -1) or current_value[part] == (-1, -1):
                            parts_acceleration[person_key][part] = -1
                        else:
                            parts_acceleration[person_key][part] += self.distance(
                                oldest_value[part], current_value[part])
                        pass
                    pass
                pass

                oldest_distance_between = self.distance(
                    oldest_features["person1"]["neck"], oldest_features["person2"]["neck"])
                current_distance_between = self.distance(
                    current_features["person1"]["neck"], current_features["person2"]["neck"])

                abuser = self.types[key]["abuser"]
                part = self.types[key]["part"]

                acceleration = parts_acceleration[abuser][part]
                if acceleration == -1:
                    values[key] = -1
                else:

                    # 兩人中心之間的距離
                    values[key]["center_distance"] = current_distance_between

                    # 兩人中心之間的距離變化
                    values[key]["center_acceleration"] = abs(
                        current_distance_between - oldest_distance_between)

                    if part[2:] == "wrist":
                        # 手跟另一個人之間的距離,如果有打到數值就會是正的
                        values[key]["wrist_distance"] = distance

                        # 手的加速度
                        values[key]["wrist_acceleration"] = acceleration

                    elif part[2:] == "ankle":
                        # 腳跟另一個人之間的距離,如果有踢到數值就會是正的
                        values[key]["ankle_distance"] = distance

                        # 腳的加速度
                        values[key]["ankle_acceleration"] = acceleration
                    pass

                    if values[key]["center_acceleration"] > self.threshold["center_acceleration"]:
                        status = 1  # violence
                    elif part[2:] == "wrist" and parts_acceleration[abuser][part] > self.threshold["wrist_acceleration"]:
                        status = 1  # violence
                    elif part[2:] == "ankle" and parts_acceleration[abuser][part] > self.threshold["ankle_acceleration"]:
                        status = 1  # violence
                    else:
                        status = 0
                    pass

                    values[key]["status"] = status
                pass
            pass
        pass
        return values
    pass

    def detect_violence_show_detail(self, history_features, closest_parts_distance):
        # 最新一筆的資料
        current_features = history_features[list(history_features)[-1]]

        # 碰觸得距離上限
        touch_distance_limit = self.calculate_touch_limit(current_features)

        # 打人的那個人所能碰觸的距離上限
        for key, value in self.types.items():
            self.types[key]["limit"] = touch_distance_limit[self.types[key]["abuser"]]
        pass
        print("===============觸碰範圍(開始)===============")
        print(closest_parts_distance)
        for key, value in closest_parts_distance.items():
            status = 0
            if value != -1:  # 有可能手沒被偵測到
                limit = self.types[key]["limit"]
                distance = value[list(value)[0]]

                if distance < limit:
                    # 最舊的資料
                    oldest_features = history_features[list(
                        history_features)[0]]

                    # 最新的資料
                    current_features = history_features[list(
                        history_features)[-1]]

                    parts_acceleration = {}

                    for (_, oldest_value), (person_key, current_value) in zip(oldest_features.items(), current_features.items()):
                        parts_acceleration[person_key] = {
                            "r_wrist": 0,
                            "l_wrist": 0,
                            "r_ankle": 0,
                            "l_ankle": 0,
                            "neck": 0
                        }

                        for part in parts_acceleration[person_key].keys():
                            if oldest_value[part] == (-1, -1) or current_value[part] == (-1, -1):
                                parts_acceleration[person_key][part] = -1
                            else:
                                parts_acceleration[person_key][part] += self.distance(
                                    oldest_value[part], current_value[part])
                            pass
                        pass
                    pass

                    oldest_distance_between = self.distance(
                        oldest_features["person1"]["neck"], oldest_features["person2"]["neck"])
                    current_distance_between = self.distance(
                        current_features["person1"]["neck"], current_features["person2"]["neck"])

                    center_acceleration = current_distance_between - oldest_distance_between
                    if center_acceleration < 0:  # 距離變近
                        status = "closer"
                    else:  # 距離變遠
                        status = "farther"
                    pass

                    abuser = self.types[key]["abuser"]
                    part = self.types[key]["part"]

                    print("tpye: {:20s}  distance: {:10.5f}  limit: {:10.5f}  兩人之間的距離: {:10.5f}  [{:7}]加速度: {:10.5f}"
                          .format(key, distance, limit, center_acceleration, part, parts_acceleration[abuser][part]), end="")

                    # print(parts_acceleration["person1"])
                    # print(parts_acceleration["person2"])

                    if center_acceleration > self.threshold["center_acceleration"]:
                        print("push")
                        status = 1
                    elif part[2:] == "wrist" and parts_acceleration[abuser][part] > self.threshold["wrist_acceleration"]:
                        print(parts_acceleration[abuser][part])
                        print("hit: ", part)
                        status = 1
                    elif part[2:] == "ankle" and parts_acceleration[abuser][part] > self.threshold["ankle_acceleration"]:
                        print("kick: ", part)
                        status = 1
                    else:
                        print()
                    pass
                pass
            pass
        pass

        print("===============觸碰範圍(結束)===============")

        if status == 1:
            return {"status": 1}
        else:
            return {"status": 0}
        pass
    pass
