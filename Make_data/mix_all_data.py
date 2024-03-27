import os
import shutil
import csv


# 上一層資料夾
previous_folder = os.path.abspath(os.path.dirname(os.getcwd()))

# 資料來源
folder_names = ["daily", "school"]

# 動作
actions = [
    ["hit", "violence"],
    ["kick", "violence"],
    ["push", "violence"],
    ["shake", "violence"],
    ["normal_single", "normal"],
    ["normal_multiple", "normal"]
]


for folder_name in folder_names:
    for action in actions:
        # 原始影片資料夾
        folder = previous_folder + "/Violence_data/remake_by_mark_" + \
            folder_name + "/" + action[0] + "/"

        # 儲存影片資料夾
        new_folder = previous_folder + \
            "/Violence_data/remake_by_mark_mix/" + action[1] + "/"
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        new_filename_index = 0
        for path in os.listdir(new_folder):
            if os.path.isfile(os.path.join(new_folder, path)):
                new_filename_index += 1

        for file in os.listdir(folder):
            new_filename_index += 1
            filename = os.path.basename(file).split('.')[0]
            video = folder + filename + ".mp4"
            video_new = new_folder + str(new_filename_index) + ".mp4"

            shutil.copy(video, video_new)
