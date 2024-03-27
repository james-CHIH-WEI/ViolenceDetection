import cv2
import os
import shutil

# 上一層資料夾
previous_folder = os.path.abspath(os.path.dirname(os.getcwd()))

statuses = ["violence", "normal"]

for status in statuses:
    # 原始影片資料夾
    folder = previous_folder + "/Violence_data/remake_by_mark_mix/" + status + "/"

    # 儲存影片資料夾
    new_folder = previous_folder + \
        "/Violence_data/remake_by_mark_mix_balance/" + status + "/"
    if not os.path.isdir(new_folder):
        os.makedirs(new_folder)

    for file in os.listdir(folder):
        # new_filename_index += 1
        filename = os.path.basename(file).split('.')[0]
        video = folder + filename + ".mp4"
        video_new = new_folder + filename + ".mp4"

        shutil.copy(video, video_new)


'''=================影片鏡像================='''
# 格式
width = 1280
height = 720
fps = 10.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")

folder = previous_folder + "/Violence_data/remake_by_mark_mix_balance/normal/"

new_filename_index = 0
for path in os.listdir(new_folder):
    if os.path.isfile(os.path.join(new_folder, path)):
        new_filename_index += 1


for file in os.listdir(folder):
    new_filename_index += 1
    filename = os.path.basename(file).split('.')[0]
    video = folder + filename + ".mp4"
    new_video = folder + str(new_filename_index) + ".mp4"
    print("run: " + video)
    cap = cv2.VideoCapture(video)

    out = cv2.VideoWriter(new_video, fourcc1, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if ret == True:
            new_frame = cv2.flip(frame, 1)  # 左右水平翻轉
            out.write(new_frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        else:
            break

    cap.release()
    out.release()


'''=================影片調亮================='''
# 格式
width = 1280
height = 720
fps = 10.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")

folder = previous_folder + "/Violence_data/remake_by_mark_mix_balance/normal/"

new_filename_index = 0
for path in os.listdir(new_folder):
    if os.path.isfile(os.path.join(new_folder, path)):
        new_filename_index += 1


for file in os.listdir(folder):
    new_filename_index += 1
    filename = os.path.basename(file).split('.')[0]
    video = folder + filename + ".mp4"
    new_video = folder + str(new_filename_index) + ".mp4"
    print("run: " + video)
    cap = cv2.VideoCapture(video)

    out = cv2.VideoWriter(new_video, fourcc1, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if ret == True:
            new_frame = frame    # 建立 output 變數

            alpha = 1.1
            beta = 30

            cv2.convertScaleAbs(frame, new_frame, alpha,
                                beta)  # 套用 convertScaleAbs

            out.write(new_frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        else:
            break

    cap.release()
    out.release()
