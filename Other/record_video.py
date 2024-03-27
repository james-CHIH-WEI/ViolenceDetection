import cv2
import os


fps = 20.0
fourcc1 = cv2.VideoWriter.fourcc("m", "p", "4", "v")
width = 1280
height = 720


# 初始化攝影機
camera = cv2.VideoCapture(2)
camera.set(cv2.CAP_PROP_FPS, fps)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# 上一層資料夾
previous_folder = os.path.abspath(os.path.dirname(os.getcwd()))

count = 0
dir = previous_folder + "/Violence_data/video"
action = "/hit"
folder = dir + action

if not os.path.isdir(folder):
    os.makedirs(folder)

for path in os.listdir(folder):
    if os.path.isfile(os.path.join(folder, path)):
        count += 1
print(count)


video_name = folder + "/" + str(count) + ".mp4"
out = cv2.VideoWriter(video_name, fourcc1, fps, (width, height))

current_frame = 0

while camera.isOpened():
    ret, frame = camera.read()
    current_frame += 1
    out.write(frame)

    cv2.putText(
        frame,
        "Frame: {}".format(current_frame),
        (40, 120),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
    )

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
out.release()
cv2.destroyAllWindows()
