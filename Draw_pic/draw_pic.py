from importlib.resources import path
import sys
sys.path.append("..")
from modules.parse_poses import parse_poses
from modules.draw import draw_poses
from modules.input_reader import VideoReader, ImageReader
import numpy as np
import cv2
from argparse import ArgumentParser

# export PYTHONPATH=../pose_extractor/build/:$PYTHONPATH
# python3 draw_pic.py --model ../model/human-pose-estimation-3d.pth --images ./pic/me.jpg


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Lightweight 3D human pose estimation demo. "
        'Press esc to exit, "p" to (un)pause video or process next image.'
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Required. Path to checkpoint with a trained model "
        "(or an .xml file in case of OpenVINO inference).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--video",
        help="Optional. Path to video file or camera id.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Optional. Specify the target device to infer on: CPU or GPU. "
        "The demo will look for a suitable plugin for device specified "
        "(by default, it is GPU).",
        type=str,
        default="GPU",
    )
    parser.add_argument(
        "--use-openvino",
        help="Optional. Run network with OpenVINO as inference engine. "
        "CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.",
        action="store_true",
    )
    parser.add_argument(
        "--use-tensorrt",
        help="Optional. Run network with TensorRT as inference engine.",
        action="store_true",
    )
    parser.add_argument(
        "--images", help="Optional. Path to input image(s).", nargs="+", default=""
    )
    parser.add_argument(
        "--height-size",
        help="Optional. Network input layer height size.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--extrinsics-path",
        help="Optional. Path to file with camera extrinsics.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fx", type=np.float32, default=-1, help="Optional. Camera focal length."
    )
    args = parser.parse_args()

    if args.video == "" and args.images == "":
        raise ValueError("Either --video or --image has to be provided")

    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO

        net = InferenceEngineOpenVINO(args.model, args.device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch

        net = InferenceEnginePyTorch(
            args.model, args.device, use_tensorrt=args.use_tensorrt
        )

    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != "":
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    stride = 8
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0

    for frame in frame_provider:
        current_time = cv2.getTickCount()
        if frame is None:
            break

        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(
            frame, dsize=None, fx=input_scale, fy=input_scale)

        scaled_img = scaled_img[
            :, 0: scaled_img.shape[1] - (scaled_img.shape[1] % stride)
        ]

        if fx < 0:
            fx = np.float32(0.8 * frame.shape[1])
        inference_result = net.infer(scaled_img)

        poses_3d, poses_2d = parse_poses(
            inference_result, input_scale, stride, fx, is_video
        )

        # ========================================================================#
        img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        draw_poses(img, poses_2d)
        draw_poses(frame, poses_2d)
        # ========================================================================#

        # draw_poses(frame, poses_2d)

        cv2.imshow("original", frame)
        cv2.imshow("new", img)

        folder = "./draw/"
        pic_name = args.images[0][5:-4]
        cv2.imwrite(folder + pic_name + "_original.jpg", frame)
        cv2.imwrite(folder + pic_name + "_new.jpg", img)

        key = cv2.waitKey(0)

        if key == esc_code:
            cv2.destroyAllWindows()
            break
