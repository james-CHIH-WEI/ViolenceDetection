import os
import cv2
import numpy as np
import pandas as pd
import joblib
from time import strftime, localtime
from sklearn import neighbors
from modules.input_reader import VideoReader
from modules.draw import draw_poses
from modules.parse_poses import parse_poses
from Subroutine.Violence import violence
import alert_message


def pre_train_model(data_folder):
    model_path = "./Export_data/" + data_folder + "/train/KNN.m"

    if not os.path.isfile(model_path):
        df = pd.read_csv("./Export_data/" + data_folder +
                         "/train/value.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

        KNN = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        KNN.fit(X, y.values.ravel())

        joblib.dump(KNN, model_path)
    else:
        KNN = joblib.load(model_path)
    return KNN


if __name__ == "__main__":
    violence = violence()

    KNN = pre_train_model("train_test_data_balance_1")

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
        net = InferenceEnginePyTorch(model, device, use_tensorrt=use_tensorrt)

    # ========================================================================= #
    stop_frist_frame = False  # 第一個畫面暫停
    stop_if_violence = False   # 判斷為暴力時暫停
    show_information = False   # 顯示畫面資訊
    detect_by_pre_model = True  # 使用預訓練模型進行辨識
    debug = False  # 進行偵錯
    alert = False  # 推播告警

    video = "./Violence_data/video_daily/hit/6.mp4"
    # ================================================================================= #
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
    current_frame = 0
    normal_status_times = 0
    temp_frames = []
    length_of_saving_frame = 15
    temp_frame_lenght_limit = 2
    video_is_saving = False  # 目前是不是在儲存影片
    alert_video_folder_image = ""
    alert_video_folder_video = ""
    alert_video_name = ""

    # 建立告警資料夾
    folder = "./Result/alert"
    if not os.path.isdir(folder):
        os.makedirs(folder)

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

        current_frame += 1
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

                result_model_predicts = []

                for feature in features.values():  # 有8組
                    if feature != -1:
                        feature_test = {
                            "center_distance": [feature["center_distance"]],
                            "center_chanege": [feature["center_chanege"]],
                            "wrist_distance": [feature["wrist_distance"]],
                            "wrist_acceleration": [feature["wrist_acceleration"]],
                            "ankle_distance": [feature["ankle_distance"]],
                            "ankle_acceleration": [feature["ankle_acceleration"]],
                        }
                        feature_test_df = pd.DataFrame(feature_test)
                        result_model_predict = KNN.predict(
                            feature_test_df)[0]
                        result_model_predicts.append(
                            result_model_predict)
                    else:
                        result_model_predicts.append(-1)
                    pass
                pass

                if 1 in result_model_predicts:
                    result_status = "violence"
                else:
                    result_status = "normal"

            previous_features = correction_features.copy()
        else:
            result_status = "normal"
            temp_frame_lenght = 0
            history_features.clear()
            previous_features.clear()

        # ============================畫面============================ #
        if show_information:
            draw_poses(frame, poses_2d)  # 畫出人體節點

            if poses_2d.shape[0] == 2:  # 如果人數是兩個人
                for i in range(2):  # 畫出人員代號
                    name = "person" + str(i + 1)
                    cv2.putText(
                        frame,
                        name,
                        (correction_features[name]["neck"][0] - 60,
                            correction_features[name]["neck"][1] - 100),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                    )

            current_time = (cv2.getTickCount() - current_time) / \
                cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05

            # fps
            cv2.putText(
                frame,
                "FPS: {}".format(int(1 / mean_time * 10) / 10),
                (40, 80),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
            )

            # frame
            cv2.putText(
                frame,
                "Frame: {}".format(current_frame),
                (40, 120),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
            )

        cv2.imshow("result", frame)
        # ======================================================================== #

        temp_frames.append(frame)

        if result_status == "violence":
            if not video_is_saving:
                current_time = str(strftime("%Y-%m-%d %H:%M:%S", localtime()))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                alert_video_folder = folder + "/" + current_time[:10]
                alert_video_folder_image = alert_video_folder + "/image"
                alert_video_folder_video = alert_video_folder + "/video"
                if not os.path.isdir(alert_video_folder):
                    os.makedirs(alert_video_folder_image)
                    os.makedirs(alert_video_folder_video)
                alert_video_name = str(current_time[11:])
                cv2.imwrite(alert_video_folder_image + "/" +
                            alert_video_name + ".png", frame)
                log_video = cv2.VideoWriter(
                    alert_video_folder_video + "/" + alert_video_name + ".mp4", fourcc, 10.0, (frame.shape[1], frame.shape[0]))

                for temp_frame in temp_frames:
                    log_video.write(np.array(temp_frame))
                temp_frames.clear()
                video_is_saving = True

        elif result_status == "normal":
            if len(temp_frames) > length_of_saving_frame:
                if video_is_saving:
                    for temp_frame in temp_frames:
                        log_video.write(np.array(temp_frame))
                    temp_frames.clear()
                    log_video.release()
                    os.system(
                        "ffmpeg -i " + alert_video_folder_video + "/" + alert_video_name + ".mp4 -vcodec libx264 -f mp4 " + alert_video_folder_video + "/" + alert_video_name + "_now.mp4  -y")
                    os.system("rm " + alert_video_folder_video +
                              "/" + alert_video_name + ".mp4")
                    os.system("mv " + alert_video_folder_video + "/" + alert_video_name +
                              "_now.mp4 " + alert_video_folder_video + "/" + alert_video_name + ".mp4")
                    if alert:
                        alert_message.push(
                            alert_video_folder_image, alert_video_folder_video, alert_video_name)

                    video_is_saving = False
                else:
                    temp_frames.pop(0)

        if debug:
            delay = 0
        else:
            # 如果判斷為暴力(暫停畫面)
            if stop_if_violence and result_status == "violence":
                delay = 0
            else:
                delay = 0
            pass

            # 第一個畫面暫停
            if stop_frist_frame:
                delay = 0
                stop_frist_frame = False
            else:
                delay = 1
            pass
        pass

        if cv2.waitKey(delay) == esc_code:
            cv2.destroyAllWindows()
            exit()
        pass
