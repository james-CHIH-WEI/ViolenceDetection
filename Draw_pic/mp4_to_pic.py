import cv2
import os

# 格式
width = 1280
height = 720
fps = 10.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")


video_name = "cut"

# 原始影片
video = "./mp4/" + video_name + ".mp4"

new_folder = "./pic/" + video_name + "/"
if not os.path.isdir(new_folder):
    os.makedirs(new_folder)

index = 0

cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if ret == True:
        new_path = new_folder + video_name + "_" + str(index) + ".jpg"
        print(new_path)
        cv2.imwrite(new_path, frame)

        cv2.imshow(video, frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        index += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()
