import cv2
import os


action = "shake"
filename = "5"

# 上一層資料夾
previous_folder = os.path.abspath(os.path.dirname(os.getcwd()))

# 原始影片
folder = previous_folder + "/Violence_data/video_daily/" + action + "/"

path = folder + filename + ".mp4"


current_frame = 0
cap = cv2.VideoCapture(path)
while True:
    ret, frame = cap.read()
    if ret == True:
        current_frame += 1

        cv2.putText(
            frame,
            "Frame: {}".format(current_frame),
            (40, 120),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
        )

        cv2.imshow(path, frame)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
