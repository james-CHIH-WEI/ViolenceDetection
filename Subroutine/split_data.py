import os
import shutil
import random
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument(
    "--folder",
    type=str,
    default="",
)

args = parser.parse_args()

original_data_folder = "remake_by_mark_mix_balance"

if args.folder == "":
    print("請輸入要儲存的資料夾名稱")
else:
    print("正在生成: ", args.folder)
    data_folder = args.folder

    action_types = ["violence", "normal"]

    for action_type in action_types:
        folder = "./Violence_data/" + original_data_folder + "/" + action_type

        new_folder = "./Violence_data/" + args.folder

        new_folder_train = new_folder + "/" + "train" + "/" + action_type
        if not os.path.isdir(new_folder_train):
            os.makedirs(new_folder_train)

        new_folder_test = new_folder + "/" + "test" + "/" + action_type
        if not os.path.isdir(new_folder_test):
            os.makedirs(new_folder_test)

        count = 0
        for path in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, path)):
                count += 1

        scale = 0.9

        X = [1] * int(count * scale)  # 訓練集
        y = [0] * (count - len(X))  # 測試集
        all = X + y
        random.shuffle(all)

        filename_index = 0
        for file in os.listdir(folder):

            filename = os.path.basename(file).split('.')[0]
            video = folder + "/" + filename + ".mp4"

            if all[filename_index] == 1:
                video_new = new_folder_train + "/" + filename + ".mp4"
            else:
                video_new = new_folder_test + "/" + filename + ".mp4"

            shutil.copy(video, video_new)

            filename_index += 1
