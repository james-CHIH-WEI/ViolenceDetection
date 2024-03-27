import cv2
import os

# 格式
width = 1280
height = 720
fps = 10.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")

# 資料夾
folder = "./temp/"
if not os.path.isdir(folder):
    os.makedirs(folder)

# 原始影片
file = "14-1"

# 儲存影片
new_file = "new_" + file

path = folder + file + ".mp4"
new_path = folder + new_file + ".mp4"

cap = cv2.VideoCapture(path)

out = cv2.VideoWriter(new_path, fourcc1, fps, (width, height))

while True:
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)

        cv2.imshow(path, frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
