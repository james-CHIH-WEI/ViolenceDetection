import cv2
import os
import csv

# 格式
width = 1280
height = 720
fps = 10.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")

# 資料來源
folder_names = ["daily", "school"]

# 動作
actions = ["hit", "kick", "push", "shake", "normal_single", "normal_multiple"]

# 上一層資料夾
previous_folder = os.path.abspath(os.path.dirname(os.getcwd()))

for folder_name in folder_names:
    for action in actions:

        # 原始影片資料夾
        folder = previous_folder + "/Violence_data/video_" + \
            folder_name + "/" + action + "/"

        # 儲存影片資料夾
        new_folder = previous_folder + "/Violence_data/remake_by_mark_" + \
            folder_name + "/" + action + "/"
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        # 要擷取的片段
        mark_file = "./mark_" + folder_name + "/" + action + ".csv"
        with open(mark_file, "r", encoding="utf-8-sig") as file:
            csvreader = csv.reader(file)
            for row in csvreader:

                filename = row[0]
                new_filename_index = 0

                path = folder + filename + ".mp4"
                print("running: " + path)
                out = ""
                cap = cv2.VideoCapture(path)
                current_frame = 0
                start_index = 1
                end_index = 2

                while True:
                    ret, frame = cap.read()
                    if ret == True:
                        current_frame += 1
                        start = row[start_index]
                        end = row[end_index]

                        if start != "" and end != "":
                            if current_frame == int(start):
                                new_filename = filename + "-" + \
                                    str(new_filename_index)
                                new_path = new_folder + "/" + new_filename + ".mp4"
                                out = cv2.VideoWriter(
                                    new_path, fourcc1, fps, (width, height))

                            if current_frame >= int(start) and current_frame <= int(end):
                                out.write(frame)

                            if current_frame == int(end):
                                start_index += 2
                                end_index += 2
                                new_filename_index += 1

                            if row[len(row) - 1] != "" and current_frame >= int(
                                row[len(row) - 1]
                            ):
                                break
                        else:
                            break

                        # cv2.imshow(path, frame)
                        key = cv2.waitKey(1)

                        if key == ord("q"):
                            break
                    else:
                        break

                cap.release()
                if out != "":
                    out.release()
                cv2.destroyAllWindows()
