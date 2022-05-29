# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import os
import sys
from pathlib import Path

from cv2 import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pynput.keyboard import Key, Controller

keyboard = Controller()
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


def loadModel(weights, half, device):
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
    return model, onnx, stride, names, pt, half, device

def runInterface(model, save_dir, device, half, conf_thres, iou_thres, classes, agnostic_nms, max_det, dataset,
                 visualize, onnx, path, img):
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    t1 = time_sync()
    if onnx:
        img = img.astype('float32')
    else:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    # Inference
    if pt:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=False, visualize=visualize)[0]
    t3 = time_sync()
    dt[1] += t3 - t2
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
    return pred, dt, seen, img

if __name__ == "__main__":
    weights = ROOT / 'yolov5s.pt'  # 模型位置

    # 数据源
    # source = 'rtmp://192.9.200.152/hls/liveview'  #网络摄像头
    source = ROOT / 'data'  # 待识别的图片或视频位置
    # source = '0' # 0 代表本地摄像头

    project = ROOT / 'runs'  # 结果保存路径
    name = 'exp'  # 结果保存文件名

    imgsz = 416  # 图片大小
    conf_thres = 0.25  # 置信度
    iou_thres = 0.45  # MNS抑制阈值
    max_det = 1000  # 每张图像的最大检测次数
    device = ''  # cuda 设备，即 0 或 0,1,2,3 或 cpu
    view_img = False  # 结果弹出展示
    save_txt = True  # 保存文本结果
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # 将识别到的物体抠出来保存图片
    nosave = False  # True 什么结果都不保存

    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 2  # 识别框的宽度像素值
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    # run()
    # 加载模型
    model, onnx, stride, names, pt, half, device = loadModel(weights, False, "")
    imgsz = 416
    # check image size
    imgsz = check_img_size(imgsz, s=stride)
    # 载入图片
    source = str(source)
    save_img = True


    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=exist_ok)  # make dir

    for path, img, im0s, vid_cap in dataset:
        pred, dt, seen, img = runInterface(model, save_dir, device, half, conf_thres, iou_thres, None, agnostic_nms,
                                           max_det, dataset, False, onnx, path, img)
        print("---------------")
        # 这个pred是非常重要的数据，是最终的结果
        # 内涵一个tensor类型的二维数组，每个小数组的最后一位是结果，其他的东西疑似坐标
        # 所有的数据全部是类似科学记数法的tensor类型
        #但是好像取出来也挺简单的？？？？？？？？？

        RNs = []
        for RN in pred[0]:
            RNs.append(int(RN[5]))
        print(RNs)

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                print('\n')
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # 人是names[0]

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #重要突破口，txt生成源码
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()

            print("识别结束")
            cv2.imshow("show", im0)
            cv2.waitKey(0)

            if save_img:
                # 这个if是做视频区别，视频的else已经删除
                if dataset.mode == 'image':
                    # save_path是动态的，注意一下
                    cv2.imwrite(save_path, im0)



