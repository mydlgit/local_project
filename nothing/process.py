import cv2
import numpy as np
import os
import glob
from PIL import Image, ImageFile
import multiprocessing
import shutil


def get_dataset_root():
    return '/home/hsy/research/car_lane/apollo'


def t1():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # png = cv2.imread('2.png')
    png = cv2.imread('/home/hsy/research/car_lane/apollo/Labels_road03/Labels/Record012/Camera 5/171206_031200232_Camera_5_bin.png')
    height, width = png.shape[:2]

    # lower是实际BGR值， upper是一个上限
    # color = [([0, 128, 128], [0, 130, 130])]           # 前进箭头
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

    bboxes = []

    for color in color_list:
        for (lower, upper) in color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(png, lower, upper)
            contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) < 1:
                continue
            contours = sorted(contours, key=lambda idx: cv2.contourArea(idx))

            x, y, w, h = cv2.boundingRect(contours[-1])      # top_left_x, top_left_y, bbox_w, bbox_h,
            bboxes.append([cv2.contourArea(contours[-1]), x, y, w, h])

    bboxes.sort(key=lambda idx: idx[0], reverse=True)
    for bbox in bboxes[:3]:
        _, x, y, w, h = bbox
        if (w >= width*0.015 and h >= height*0.015 and y <= 0.75*height) or \
                (w >= width*0.035 and h >= height*0.035 and 0.75*height < y <= 0.78*height):

            print(x, y, w, h)
            cv2.rectangle(png, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('img', png)
    cv2.waitKey(0)
    # cv2.imwrite('post.png', png)


# 统计 apollo数据集中有多少张包含符合要求的箭头的图片
def t2(seq_th):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    names_set = set()
    root = '/home/hsy/research/car_lane/apollo'

    color_list = [
        [([0, 128, 128], [0, 130, 130])],
        [([100, 100, 150], [100, 105, 155])],
        [([160, 78, 128], [160, 80, 130])],
        [([230, 0, 0], [230, 2, 2])],
        [([180, 165, 180], [180, 165, 180])],
        [([35, 142, 107], [35, 145, 110])]
    ]
    arrow = ['forward', 'forward+turn right', 'forward+turn left', 'turn around left', 'turn left', 'turn right']

    cnt = [0, 0, 0, 0, 0, 0]

    labels_ = 'Labels_road0%d' % seq_th
    color_img_path = os.path.join(root, labels_)
    globs = glob.glob(color_img_path+'/*/*/*/*')
    total = len(globs)
    for idx, img_path in enumerate(globs):
        img = cv2.imread(img_path)

        try:
            height, width = img.shape[:2]
        except:
            pil_img = Image.open(img_path)
            pil_img.save(img_path)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

        for color_idx, color in enumerate(color_list):
            for (lower, upper) in color:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(img, lower, upper)
                contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    contours = sorted(contours, key=lambda xx: cv2.contourArea(xx), reverse=True)
                    x, y, w, h = cv2.boundingRect(contours[-1])

                    if (w >= width * 0.015 and h >= height * 0.015 and y <= 0.75 * height) or \
                            (w >= width * 0.035 and h >= height * 0.035 and 0.75 * height < y <= 0.78 * height):
                        cnt[color_idx] += 1
                        names_set.add(img_path+'\n')
        print('\r'+labels_+': %d of %d has finished' % (idx+1, total), end='')

    print('\nAfter '+labels_)
    for num, name in zip(cnt, arrow):
        print(name+':%d' % num)

    names_set = list(names_set)
    with open('%s_arrow.txt' % labels_, 'w') as f:
        f.writelines(names_set)

    # return labels_+' finish'


def t2_multi_process():
    color_img_seq = [2, 3, 4]
    process_pool = multiprocessing.Pool(3)
    for i in color_img_seq:
        process_pool.apply_async(t2, (i,))
    process_pool.close()
    process_pool.join()


# 对图片中的箭头进行反转
def imgfusion(A, B, m):
    GA = A.copy().astype(np.float32)
    GB = B.copy().astype(np.float32)
    GMA = m.copy().astype(np.float32)
    GMB = 1 - GMA
    length = max(int(max(GA.shape) / 50) * 2 + 1, 3)
    GMA = cv2.GaussianBlur(GMA, (length, length), 0)
    GMB = cv2.GaussianBlur(GMB, (length, length), 0)

    outimg = GA * GMA + GB * GMB
    return outimg


def t3(img_path, img_save_root, label_save_root):
    label_path = img_path.replace('ColorImage', 'Labels').replace('.jpg', '_bin.png')

    jpgname = os.path.split(img_path)[1]
    png = cv2.imread(label_path)
    jpg = cv2.imread(img_path)
    height, width = jpg.shape[:2]

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

    bboxes = []

    for color in color_list:
        for (lower, upper) in color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(png, lower, upper)
            contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) < 1:
                continue
            contours = sorted(contours, key=lambda idx: cv2.contourArea(idx))

            x, y, w, h = cv2.boundingRect(contours[-1])  # top_left_x, top_left_y, bbox_w, bbox_h,
            bboxes.append([cv2.contourArea(contours[-1]), x, y, w, h])

    bboxes.sort(key=lambda idx: idx[0], reverse=True)
    best_x = -1
    for bbox in bboxes[:3]:
        _, x, y, w, h = bbox
        mean_x = (2 * x + w) / 2
        if best_x < mean_x:
            best_x = mean_x
            best_bbox = bbox

    _, x, y, w, h = best_bbox

    y_line_right = png[y, x + w:-1, :]
    y_line_left = png[y, 0:x, :]
    rslt = np.where((y_line_left[:, 0] == 180) & (y_line_left[:, 1] == 130) & (y_line_left[:, 2] == 70))
    left = min(rslt[0])
    top_left = [left, y]

    rslt = np.where((y_line_right[:, 0] == 180) & (y_line_right[:, 1] == 130) & (y_line_right[:, 2] == 70))
    right = max(rslt[0])
    top_right = [x + w + right, y]

    y_line_right = png[y + h, x + w:-1, :]
    y_line_left = png[y + h, 0:x, :]
    rslt = np.where((y_line_left[:, 0] == 180) & (y_line_left[:, 1] == 130) & (y_line_left[:, 2] == 70))
    left = min(rslt[0])
    bottom_left = [left, y + h]

    rslt = np.where((y_line_right[:, 0] == 180) & (y_line_right[:, 1] == 130) & (y_line_right[:, 2] == 70))
    right = max(rslt[0])
    bottom_right = [x + w + right, y + h]

    length = max(w, h)
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    dst = np.array([[0, 0], [length, 0], [length, length], [0, length]], dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(jpg, M, (length, length))
    warped = cv2.flip(warped, 0)

    inv_warped = cv2.warpPerspective(warped, inv_M, (jpg.shape[1], jpg.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    mask_3c = np.zeros((height, width, 3), dtype=np.float32)
    mask_3c[y:y+h, x:x+w] = 1

    bgr_out = imgfusion(inv_warped.astype(np.float32), jpg.astype(np.float32), mask_3c).astype(np.uint8)

    cv2.imwrite(os.path.join(img_save_root, jpgname), bgr_out)
    fid = open(os.path.join(label_save_root, jpgname.replace('.jpg', '.txt')), 'w')
    cx, cy, w, h = (x+w)/2/width, (y+h)/2/height, w/width, h/height
    fid.write('%d %.8f %.8f %.8f %.8f\n' % (1, cx, cy, w, h))
    fid.close()


# 移动有箭头的图片到指定文件下
def t4():
    arrows = ['Labels_road0%d_arrow.txt' % x for x in range(2, 5)]
    root = get_dataset_root()
    save_root = '/home/hsy/research/car_lane/car_sign/images/'
    seg_save_root = '/home/hsy/research/car_lane/car_sign/seg_images/'

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(seg_save_root, exist_ok=True)

    for arrow_txt in arrows:
        with open(arrow_txt, 'r') as f:
            lines = [x.strip().split('apollo')[1] for x in f.read().splitlines()]

        for img_path in lines:
            img_path = os.path.join(root, img_path)
            shutil.copy(src=img_path, dst=seg_save_root)

            color_img_path = img_path.replace('Labels', 'ColorImage').replace('_bin.png', '.jpg')
            shutil.copy(color_img_path, save_root)


if __name__ == '__main__':
    print()
    # t1()
    # t2_multi_process()





