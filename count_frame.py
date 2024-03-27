import os

import cv2

from modules.input_reader import VideoReader

# folder = ["remake_by_mark_mix"]
# action_or_types = ["violence", "normal"]

folders = ["video_daily", "video_school"]
action_or_types = ["hit", "kick", "push", "shake",
                   "weapon", "normal_single", "normal_multiple"]
count_frame = 0

for folder in folders:
    for action_or_type in action_or_types:
        folder_count_frame = "./violence_data/" + folder + "/" + action_or_type

        for file in os.listdir(folder_count_frame):
            filename = os.path.basename(file).split('.')[0]
            video = folder_count_frame + "/" + filename + ".mp4"
            print("執行檔案: ", video)

            frame_provider = VideoReader(video)
            esc_code = 27

            for frame in frame_provider:
                count_frame += 1
                key = cv2.waitKey(1)
                if key == esc_code:
                    exit()

print(count_frame)
