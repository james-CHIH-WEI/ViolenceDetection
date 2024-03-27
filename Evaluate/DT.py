import os
import csv
import pandas as pd
import joblib
from argparse import ArgumentParser
from sklearn import tree
from Subroutine.Violence import violence


def pre_train_model(data_folder):
    model_path = "./Export_data/" + data_folder + "/train/DT.m"

    if not os.path.isfile(model_path):
        df = pd.read_csv("./Export_data/" + data_folder + "/train/value.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

        DT = tree.DecisionTreeClassifier(random_state=42)
        DT.fit(X, y.values.ravel())

        joblib.dump(DT, model_path)
    else:
        DT = joblib.load(model_path)
    return DT


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
        model_name = "DT"
        print("模型:" + model_name + "  讀取資料夾: ", args.folder)

        data_folder = args.folder

        violence = violence()

        DT = pre_train_model(data_folder)

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
        # 種類
        action_types = ["violence", "normal"]

        for action_type in action_types:
            count_all = 0
            count_true = 0

            # csv格式的影片資料夾
            folder_original_data = (
                "./Violence_data/" + data_folder + "/test/" + action_type + "_csv"
            )

            # 儲存辨識結果的資料夾
            folder_result = "./Result/" + data_folder + "/"
            folder_result_single = folder_result + "single/"
            if not os.path.isdir(folder_result_single):
                os.makedirs(folder_result_single)

            # 每個影片的辨識結果
            result_action_type_csv = (
                folder_result_single + model_name + "_" + action_type + ".csv"
            )
            with open(result_action_type_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["video", "status"])

            result_all_csv = folder_result + "/evaluate.csv"
            if not os.path.isfile(result_all_csv):
                with open(result_all_csv, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ["Data", "Model", "Action", "All",
                            "True", "False", "Accuracy"]
                    )

            for file in os.listdir(folder_original_data):
                result_status = "normal"
                count_all += 1
                filename = os.path.basename(file).split(".")[0]
                video = folder_original_data + "/" + filename + ".csv"
                print("執行檔案: {:70s}".format(video), end="  ")

                df = pd.read_csv(video)

                for index, result in df.iterrows():
                    result_model_predicts = []
                    result_test = {
                        "center_distance": [result["center_distance"]],
                        "center_chanege": [result["center_chanege"]],
                        "wrist_distance": [result["wrist_distance"]],
                        "wrist_acceleration": [result["wrist_acceleration"]],
                        "ankle_distance": [result["ankle_distance"]],
                        "ankle_acceleration": [result["ankle_acceleration"]],
                    }
                    result_test_df = pd.DataFrame(result_test)
                    result_model_predict = DT.predict(result_test_df)[0]
                    result_model_predicts.append(result_model_predict)

                    # print(result_model_predicts)
                    if 1 in result_model_predicts:
                        result_status = "violence"
                    else:
                        result_status = "normal"

                    if result_status == "violence":
                        break

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
