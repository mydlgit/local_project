import os
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
import glob
import cv2
import numpy as np


DEVICE = torch.device('cuda:0')
img_size = 512


def detect_car_tail():
    model_path = '../weights/yolov5x_car_tail.pt'
    model = attempt_load(model_path, map_location=DEVICE)  # load FP32 model
    model.half()

    dataset = LoadImages(path='/home/hsy/research/car_lane/car_sign/images/', img_size=img_size)
    label_save = '/home/hsy/research/car_lane/car_sign/labels/'
    os.makedirs(label_save, exist_ok=True)

    img = torch.zeros((1, 3, img_size, img_size), device=DEVICE)  # init img
    _ = model(img.half())                                         # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(DEVICE)
        img = img.half()
        img /= 255.0                                              # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=1, agnostic=True)

        txt_path = label_save+'%s.txt' % os.path.split(path)[1].split('.')[0]
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)

                for *xyxy, conf, cls in reversed(det):
                    # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format


def arrow_labeling():
    seg_save_root = '/home/hsy/research/car_lane/car_sign/seg_images/'

    # lower是实际BGR值， upper是一个上限
    # color = [([0, 128, 128], [0, 130, 130])]         # 前进箭头
    # color = [([100, 100, 150], [100, 105, 155])]     # 前进+右转箭头
    # color = [([160, 78, 128], [160, 80, 130])]       # 前进+左转箭头
    # color = [([230, 0, 0], [230, 2, 2])]             # 向左调头
    # color = [([180, 165, 180], [180, 165, 180])]     # 左转弯箭头
    # color = [([35, 142, 107], [35, 145, 110])]       # 右转弯箭头

    color_list = [
        [([0, 128, 128], [0, 130, 130])],
        [([100, 100, 150], [100, 105, 155])],
        [([160, 78, 128], [160, 80, 130])],
        [([230, 0, 0], [230, 2, 2])],
        [([180, 165, 180], [180, 165, 180])],
        [([35, 142, 107], [35, 145, 110])]
    ]

    globs = glob.glob(seg_save_root+'*')
    total = len(globs)
    for ix, img_path in enumerate(globs):
        seg_img = cv2.imread(img_path)
        height, width = seg_img.shape

        bboxes = []
        for color_idx, color in enumerate(color_list):
            for (lower, upper) in color:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(seg_img, lower, upper)
                contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) < 1:
                    continue

                contours_areas = np.array([cv2.contourArea(x) for x in contours])
                area_sort_idx = contours_areas.argsort()    # ascendent
                contours = contours[area_sort_idx]

                x, y, w, h = cv2.boundingRect(contours[0])  # top_left_x, top_left_y, bbox_w, bbox_h,
                bboxes.append([cv2.contourArea(contours[0]), x, y, w, h, color_idx])

                x, y, w, h = cv2.boundingRect(contours[1])  # top_left_x, top_left_y, bbox_w, bbox_h,
                bboxes.append([cv2.contourArea(contours[1]), x, y, w, h, color_idx])

        bboxes.sort(key=lambda idx: idx[0], reverse=True)
        lines = []
        for bbox in bboxes[:3]:
            _, x, y, w, h, color_idx = bbox
            cx, cy = (x+w/2)/height, (y+h/2)/height
            w, h = w/width, h/height
            lines.append('%d %.8f %.8f %.8f %.8f\n' % (color_idx+1, cx, cy, w, h))

        with open(img_path.replace('seg_images', 'labels').replace('_bin.png', '.txt'), 'a') as f:
            f.writelines(lines)

        print('\r%d of %d has finished' % (ix+1, total), end='')


if __name__ == '__main__':
    print()
