import cv2
import os

# 格式
width = 1280
height = 720
fps = 10.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")

# 資料夾
folder = "./pic/"
if not os.path.isdir(folder):
    os.makedirs(folder)

# 原始影片
file = "13"

path = "./mp4/" + file + ".mp4"
new_path = folder + file + ".jpg"

cap = cv2.VideoCapture(path)

print(path)

while True:
    ret, frame = cap.read()
    if ret == True:

        cv2.imshow(path, frame)

        key = cv2.waitKey(0)

        if key == ord("s"):
            cv2.imwrite(new_path, frame)
        elif key == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
