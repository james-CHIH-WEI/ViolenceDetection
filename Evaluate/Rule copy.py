import os
import cv2
import numpy as np
import csv
from argparse import ArgumentParser
from modules.input_reader import VideoReader
from modules.parse_poses import parse_poses
from Subroutine.Violence import violence

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--folder", type=str, default="",
    )

    args = parser.parse_args()

    if args.folder == "":
        print("請輸入要讀取的資料夾")
    else:
        # 模型的名稱
        model_name = "Rule"
        print("模型:" + model_name + "  讀取資料夾: ", args.folder)

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

        # 種類
        action_types = ["violence", "normal"]

        for action_type in action_types:
            count_all = 0
            count_true = 0
            folder_original_data = "./Violence_data/" + folder_name + "/" + action_type

            # 儲存辨識結果的資料夾
            folder_result = "./Result/" + folder_name + "/"
            if not os.path.isdir(folder_result):
                os.makedirs(folder_result)

            # 每個影片的辨識結果
            result_action_type_csv = (
                folder_result + model_name + "_" + action_type + "_v2.csv"
            )
            with open(result_action_type_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["video", "status"])

            result_all_csv = folder_result + "/result_new_v2.csv"
            if not os.path.isfile(result_all_csv):
                with open(result_all_csv, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ["Data", "Model", "Action", "All",
                            "True", "False", "Accuracy"]
                    )

            for file in os.listdir(folder_original_data):
                count_all += 1
                filename = os.path.basename(file).split(".")[0]
                video = folder_original_data + "/" + filename + ".mp4"
                print("執行檔案: {:70s}".format(video), end="  ")
                new_file_index = 0

                frame_provider = VideoReader(video)
                base_height = height_size
                stride = 8
                delay = 1
                esc_code = 27
                mean_time = 0

                result_status = "normal"
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
                        frame, dsize=None, fx=input_scale, fy=input_scale
                    )

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
                            poses_2d
                        )

                        if len(previous_features) == 0:  # 如果沒有過去的資料
                            correction_features = current_features.copy()
                        else:
                            # 人員校正（因為沒一張影格都是重新偵測的,所以不知道誰是誰）
                            correction_features = violence.identify_person(
                                previous_features, current_features
                            )

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
                                "l_hip": correction_features[name]["l_hip"],
                            }

                        # 當數量達到指定的數量時
                        if len(history_features) == temp_frame_lenght_limit:
                            # 計算指定部位（手,腳）與另一個人的 "最近的節點" 距離
                            closest_parts_distance = violence.calculate_closest_parts_distance(
                                correction_features
                            )

                            results = violence.detect_violence_return_value_new_v2(
                                history_features, closest_parts_distance
                            )

                            result_model_predicts = []

                            for result in results.values():  # 有8組
                                if result != -1:
                                    result_model_predicts.append(
                                        result["status"])
                                else:
                                    result_model_predicts.append(-1)

                            if 1 in result_model_predicts:
                                result_status = "violence"
                            else:
                                result_status = "normal"

                        previous_features = correction_features.copy()
                    else:
                        result_status = "normal"
                        temp_frame_lenght = 0
                        previous_features.clear()

                    if result_status == "violence":
                        break

                    key = cv2.waitKey(1)

                    if key == esc_code:
                        cv2.destroyAllWindows()
                        exit()

                print(
                    "real: {:10s} predict: {:10s}".format(
                        action_type, result_status),
                    end=" ",
                )

                if result_status == action_type:
                    print("True")
                    count_true += 1
                else:
                    print("False")

                with open(result_action_type_csv, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([video, result_status])

            with open(result_all_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if action_type == "violence":
                    value = [
                        data_folder,
                        model_name,
                        action_type,
                        count_all,
                        count_true,
                        count_all - count_true,
                        "{:.2f}%".format(count_true / count_all * 100),
                    ]
                else:
                    value = [
                        data_folder,
                        model_name,
                        action_type,
                        count_all,
                        count_all - count_true,
                        count_true,
                        "{:.2f}%".format(count_true / count_all * 100),
                    ]
                writer.writerow(value)
