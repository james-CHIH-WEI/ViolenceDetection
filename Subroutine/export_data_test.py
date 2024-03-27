import os

import cv2
import numpy as np
import csv

from argparse import ArgumentParser
from modules.input_reader import VideoReader
from modules.parse_poses import parse_poses

from Violence import violence

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        default="",
    )

    args = parser.parse_args()

    if args.folder == "":
        print("請輸入要讀取的資料夾")
    else:
        print("讀取資料夾: ", args.folder)
        data_folder = args.folder

        violence = violence()

        type = "gpu"

        if type == "gpu":
            model = "./model/human-pose-estimation-3d.pth"
            device = "GPU"
            use_openvino = False
        elif type == "cpu":
            model = "./model/human-pose-estimation-3d.xml"
            device = "CPU"
            use_openvino = True

        use_tensorrt = None
        extrinsics_path = None  # default
        height_size = 256  # default
        fx = -1  # default

        if use_openvino:
            from modules.inference_engine_openvino import InferenceEngineOpenVINO
            net = InferenceEngineOpenVINO(model, device)
        else:
            from modules.inference_engine_pytorch import InferenceEnginePyTorch
            net = InferenceEnginePyTorch(
                model, device, use_tensorrt=use_tensorrt)

        # ========================================================================= #

        # 資料來源
        folder_name = data_folder + "/test"

        actions = ["violence", "normal"]

        for action in actions:

            # 影片的資料夾
            folder_original_data = "./Violence_data/" + folder_name + "/" + action

            # 匯出的資料夾
            export_data = "./Violence_data/" + folder_name + "/" + action + "_csv"
            if not os.path.isdir(export_data):
                os.makedirs(export_data)

            for file in os.listdir(folder_original_data):
                filename = os.path.basename(file).split('.')[0]

                value_csv = export_data + "/" + filename + ".csv"
                with open(value_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ["center_distance",
                            "center_acceleration",
                            "wrist_distance",
                            "wrist_acceleration",
                            "ankle_distance",
                            "ankle_acceleration",
                         ])

                video = folder_original_data + "/" + filename + ".mp4"
                print("執行檔案: ", video, end="  ")

                frame_provider = VideoReader(video)
                base_height = height_size
                stride = 8
                delay = 1
                esc_code = 27
                mean_time = 0

                current_features = {}
                previous_features = {}
                history_features = {}
                temp_frame_lenght = 0
                temp_frame_lenght_limit = 2
                video_saving = False

                for frame in frame_provider:
                    current_time = cv2.getTickCount()
                    if frame is None:
                        break
                    input_scale = base_height / frame.shape[0]
                    scaled_img = cv2.resize(
                        frame, dsize=None, fx=input_scale, fy=input_scale)

                    scaled_img = scaled_img[
                        :, 0: scaled_img.shape[1] - (scaled_img.shape[1] % stride)
                    ]

                    if fx < 0:
                        fx = np.float32(0.8 * frame.shape[1])
                    inference_result = net.infer(scaled_img)

                    poses_3d, poses_2d = parse_poses(
                        inference_result, input_scale, stride, fx, True
                    )

                    # ------------------------------ 判斷 ------------------------------ #
                    if poses_2d.shape[0] == 2:  # 人數為兩個人的時候才進判斷
                        temp_frame_lenght += 1

                        # 轉換成自訂的格式
                        current_features = violence.convert_poses_2d_to_features(
                            poses_2d)

                        if len(previous_features) == 0:  # 如果沒有過去的資料
                            correction_features = current_features.copy()
                        else:
                            # 人員校正（因為沒一張影格都是重新偵測的,所以不知道誰是誰）
                            correction_features = violence.identify_person(
                                previous_features, current_features)

                        history_features[temp_frame_lenght] = {}
                        # 如果大於指定的數量就把最舊的刪掉
                        if len(history_features) > temp_frame_lenght_limit:
                            history_features.pop(list(history_features)[0])

                        for i in range(2):
                            name = "person" + str(i + 1)

                            history_features[temp_frame_lenght][name] = {
                                "r_wrist": correction_features[name]["r_wrist"],
                                "l_wrist": correction_features[name]["l_wrist"],
                                "r_ankle": correction_features[name]["r_ankle"],
                                "l_ankle": correction_features[name]["l_ankle"],
                                "neck": correction_features[name]["neck"],
                                "r_hip": correction_features[name]["r_hip"],
                                "l_hip": correction_features[name]["l_hip"]
                            }

                        # 當數量達到指定的數量時
                        if len(history_features) == temp_frame_lenght_limit:
                            # 計算指定部位（手,腳）與另一個人的 "最近的節點" 距離
                            closest_parts_distance = violence.calculate_closest_parts_distance(
                                correction_features)

                            features = violence.calculate_features(
                                history_features, closest_parts_distance)

                            for feature in features.values():  # 有8組
                                if feature != -1:
                                    with open(value_csv, 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile)
                                        writer.writerow(
                                            [
                                                feature["center_distance"],
                                                feature["center_acceleration"],
                                                feature["wrist_distance"],
                                                feature["wrist_acceleration"],
                                                feature["ankle_distance"],
                                                feature["ankle_acceleration"],
                                            ]
                                        )

                        previous_features = correction_features.copy()
                    else:
                        temp_frame_lenght = 0
                        previous_features.clear()

                    key = cv2.waitKey(0)

                    if key == esc_code:
                        cv2.destroyAllWindows()
                        exit()

                cv2.destroyAllWindows()
                print()
