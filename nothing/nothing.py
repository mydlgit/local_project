import os
import glob
import shutil
import cv2
from utils.general import kmean_anchors
import numpy as np
import json
import torch
import yaml


# 将原始车尾数据集中的image和label分开
def t1():
    root = '/home/hsy/research/car_lane/car_tail/images/'
    save_root = '/home/hsy/research/car_lane/car_tail/labels/'
    os.makedirs(save_root, exist_ok=True)

    for i, file in enumerate(glob.glob(root + '*.txt')):
        shutil.copy(file, save_root)
        os.unlink(file)
        print('\r%d has finished' % (i+1), end='')


# 构建一个小的数据集， 用于计算anchor
def t2():
    img_root = '/home/hsy/research/car_lane/TT100K/data/xunlian/images/train/'
    label_root = img_root.replace('images', 'labels')

    img128 = '/home/hsy/research/car_lane/TT100K/data/data256/images/'
    label128 = img128.replace('images', 'labels')
    os.makedirs(img128, exist_ok=True)
    os.makedirs(label128, exist_ok=True)

    img_list = os.listdir(img_root)
    np.random.shuffle(img_list)
    num = 256
    for i, name in enumerate(img_list[:num]):
        shutil.copy(src=img_root+name, dst=img128)
        shutil.copy(src=label_root+name.replace('.png', '.txt').replace('.jpg', '.txt'), dst=label128)
        print('\r %d of %d has finished' % (i+1, num), end=' ')


# bbox可视化
def t3():
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(162)]
    # colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
    name = 'TT100K/data'
    save_root = '/home/hsy/research/car_lane/%s/data256/visualize/' % name
    os.makedirs(save_root, exist_ok=True)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    img_list = glob.glob('/home/hsy/research/car_lane/%s/data256/images/*' % name)

    for img_path in img_list:
        txt_path = img_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('images', 'labels')
        try:
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
        except:
            print(img_path)

        with open(txt_path, 'r') as f:
            labels = f.readlines()

        for i, label in enumerate(labels):
            array = label.strip().split(' ')
            label = int(array[0])
            x, y, w, h = map(float, array[1:5])
            x, y, w, h = int(width*x), int(height*y), int(width*w), int(height*h)
            c1, c2 = (x - w // 2, y - h // 2), (x+w//2, y+h//2)

            tl = round(0.002 * (height + width) / 2) + 1
            cv2.rectangle(img, c1, c2, color=colors[label], thickness=tl)

        cv2.imwrite(save_root+os.path.split(img_path)[1], img)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)


# 计算anchor
def t4():
    result = kmean_anchors(path='../data/TT100K256.yaml', img_size=512)


# 划分车尾数据集
def t5():
    img_root = '/home/hsy/research/car_lane/car_tail/images/'
    label_root = '/home/hsy/research/car_lane/car_tail/labels/'

    save_img_root = '/home/hsy/research/car_lane/car_tail/train/images/'
    save_label_root = '/home/hsy/research/car_lane/car_tail/train/labels/'
    os.makedirs(save_img_root, exist_ok=True)
    os.makedirs(save_label_root, exist_ok=True)

    img_list = os.listdir(img_root)
    number = len(img_list)

    train = int(0.85*number)

    train_images = img_list[:train]
    val_images = img_list[train:]

    for img_name in train_images:
        shutil.copy(src=img_root+img_name, dst=save_img_root)
        shutil.copy(src=label_root+img_name.replace('.jpg', '.txt'), dst=save_label_root)

    save_img_root = save_img_root.replace('train', 'val')
    save_label_root = save_label_root.replace('train', 'val')
    os.makedirs(save_img_root, exist_ok=True)
    os.makedirs(save_label_root, exist_ok=True)
    for img_name in val_images:
        shutil.copy(src=img_root + img_name, dst=save_img_root)
        shutil.copy(src=label_root + img_name.replace('.jpg', '.txt'), dst=save_label_root)


# 筛选标签错误的
def t6():
    label_root = '/home/hsy/research/car_lane/car_tail/xunlian/labels/train/'
    unused = '/home/hsy/research/car_lane/car_tail/xunlian/unused/'
    error = []

    for i, txt in enumerate(glob.glob(label_root+'*')):
        with open(txt, 'r') as f:
            labels = f.readlines()

        labels = [x.strip().split(' ')[1:5] for x in labels]
        labels = np.array(labels, dtype=np.float32)
        if not (labels <= 1).all():
            print(txt)
            error.append(txt+'\n')
            shutil.move(src=txt.replace('labels', 'images').replace('.txt', '.jpg'), dst=unused)
            os.unlink(txt)

        print('\r%d has finished' % (i+1), end='')


# 对于单目标检测的模型重新加载类别name
def t7():
    name = 'yolov5x_car_tail'
    weights = torch.load('../weights/%s.pt' % name, map_location='cuda')
    model = weights['model']
    with open('../data/car_tail.yaml') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']
    model.names = names

    torch.save(weights, '../weights/%s.pt' % name)


# CCTSDB 训练集制作
def t8():
    with open('/home/hsy/research/car_lane/CCTSDB/GroundTruth/GroundTruth.txt', 'r') as f:
        labels = f.readlines()
    label_path = '/home/hsy/research/car_lane/CCTSDB/xunlian/labels/train/'
    if os.path.exists(label_path):
        shutil.rmtree(label_path)
    os.makedirs(label_path, exist_ok=True)

    img_path = '/home/hsy/research/car_lane/CCTSDB/xunlian/images/train/'

    label_dict = {'warning': '0', 'prohibitory': '1', 'mandatory': '2'}
    img_label = dict()
    for i, label in enumerate(labels):
        info = label.strip().split(';')
        img_name = info[0]
        if img_name in img_label.keys():
            img_label[img_name].append(info[1:])
        else:
            img_label[img_name] = [info[1:]]

    for i, path in enumerate(glob.glob(img_path+'*')):
        img = cv2.imread(path)
        img_name = os.path.split(path)[1]
        height, width = img.shape[:2]

        lines = []
        if img_name in img_label.keys():
            for info in img_label[img_name]:
                x1, y1, x2, y2 = info[0:4]
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                cx, cy = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1+1.0, y2-y1+1.0
                cls = label_dict[info[4]]
                line = '%s %.8f %.8f %.8f %.8f\n' % (cls, cx/width, cy/height, w/width, h/height)
                lines.append(line)

        with open(label_path+img_name.replace('.png', '.txt'), 'w') as f1:
            f1.writelines(lines)

        print('\r%d of %d has finished' % (i+1, len(glob.glob(img_path+'*'))), end='')


# CCTSDB测试集制作
def t9():
    labels = glob.glob('/home/hsy/research/car_lane/CCTSDB/test/*.txt')
    imgs_path = '/home/hsy/research/car_lane/CCTSDB/xunlian/images/test/'
    save_path = imgs_path.replace('images', 'labels')
    os.makedirs(save_path, exist_ok=True)

    for label in labels:
        label_name = os.path.split(label)[1]
        img = cv2.imread(imgs_path+label_name.replace('.txt', '.png'))
        height, width = img.shape[:2]

        with open(label, 'r') as f:
            info = f.readlines()
        lines = []
        for line in info:
            array = line.strip().split(' ')
            cls = array[0]
            if cls == '1':
                cls = '2'
            elif cls == '2':
                cls = '1'
            x1, y1, x2, y2 = map(float, array[1:])
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1+1, y2-y1+1
            lines.append('%s %.8f %.8f %.8f %.8f\n' % (cls, cx/width, cy/height, w/width, h/height))

        with open(save_path+label_name, 'w') as f:
            f.writelines(lines)


# 筛选TT100K中标签数量大于100的label, 以及可视化所有的sign
def t10(visual=False):
    img_root = '/home/hsy/research/car_lane/TT100K/data/xunlian/images/'
    save_root = '/home/hsy/research/car_lane/TT100K/data/marks/signs/'
    jsonfile = json.load(open('annotations.json', 'r'))
    imgs = jsonfile['imgs']
    typedict = dict()
    empty = 0
    for img_name in imgs.keys():
        imginfo = imgs[img_name]

        if 'other' in imginfo['path']:
            continue

        if len(imginfo['objects']) > 1:
            objects = imginfo['objects']
            for cat in objects:
                if cat['category'] not in typedict.keys():
                    typedict[cat['category']] = 1

                    if visual:
                        bbox = cat['bbox']
                        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(
                            bbox['ymax'])
                        img = cv2.imread(img_root+imginfo['path'].replace('test', 'val'))
                        try:
                            cv2.imwrite(save_root+'%s.jpg' % cat['category'], img[ymin:ymax, xmin:xmax])
                        except:
                            print()

                else:
                    typedict[cat['category']] += 1

    sorted_dict = sorted(typedict.items(), key=lambda x: x[1], reverse=True)
    all_keys = [x[0] for x in sorted_dict]
    print(all_keys)
    print(len(all_keys))
    need = []
    for (label, num) in sorted_dict:
        if num > 100:
            need.append(label)

    print(len(need), need)
    # # ['pn', 'pne', 'i5', 'p11', 'pl40', 'po', 'pl50', 'io', 'pl80', 'i4', 'p26', 'pl60', 'pl100', 'il60', 'pl5',
    # #      'pl30', 'i2', 'w57', 'il80', 'pl120', 'ip', 'p10', 'p5', 'p12', 'p3', 'p23', 'w55', 'pl70', 'il100', 'w13',
    # #      'p27', 'ph4.5', 'pl20', 'pm55', 'ph4', 'p6']


# 将TT100K annotation.json转换为各自对应的label文件
def t11():
    # 按数量排序
    labels = ['pn', 'pne', 'i5', 'p11', 'pl40', 'po', 'pl50', 'io', 'pl80', 'i4', 'p26', 'pl60', 'pl100', 'il60', 'i2',
              'pl30', 'pl5', 'w57', 'il80', 'pl120', 'ip', 'p10', 'p5', 'p12', 'p3', 'p23', 'w55', 'pl70', 'il100',
              'p27', 'ph4.5', 'pl20', 'w13', 'pm55', 'ph4', 'p6', 'ph5', 'wo', 'w59', 'pg', 'pm20', 'il90', 'pr40',
              'p19', 'pm30', 'w32', 'pb', 'pa14', 'w30', 'w58', 'p9', 'p25', 'pl15', 'pl90', 'p1', 'w63', 'p18', 'i10',
              'p22', 'p14', 'il50', 'w22', 'pl110', 'ps', 'p17', 'pl10', 'ph4.2', 'pa13', 'ph3', 'il110', 'pr60', 'p16',
              'w3', 'w21', 'pr20', 'w47', 'i13', 'pr50', 'pw3.2', 'il70', 'pm10', 'p2', 'p8', 'pr30', 'ph2.2', 'i12',
              'ph2.5', 'pm15', 'w45', 'pw4', 'ph3.5', 'w20', 'w16', 'w46', 'i1', 'w18', 'w15', 'pl25', 'ph2', 'w42',
              'pm2', 'w41', 'p13', 'pa10', 'pm35', 'ph4.8', 'pr70', 'p15', 'pl35', 'ph4.3', 'i3', 'i14', 'pw3', 'pm5',
              'p4', 'w34', 'pm40', 'pm50', 'ph2.8', 'pr80', 'pa12', 'w35', 'i11', 'pw4.5', 'i15', 'w24', 'w10', 'pr45',
              'p28', 'ph2.9', 'ph3.2', 'p21', 'p20', 'ph1.5', 'w12', 'w37', 'pr100', 'pw3.5', 'pm8', 'w38', 'ph2.4',
              'pl0', 'pm2.5', 'w8', 'ph5.3', 'pa8', 'w28', 'ph2.1', 'w66', 'p24', 'w5', 'pw2.5', 'pl3', 'w43', 'w56',
              'pr10', 'pm13', 'pw4.2', 'pw2', 'ph5.5', 'w2', 'ph3.8']

    jsonfile = json.load(open('annotations.json', 'r'))
    imgs = jsonfile['imgs']

    test_label_root = '/home/hsy/research/car_lane/TT100K/data/xunlian/ori_labels/val/'
    train_label_root = '/home/hsy/research/car_lane/TT100K/data/xunlian/ori_labels/train/'
    os.makedirs(test_label_root, exist_ok=True)
    os.makedirs(train_label_root, exist_ok=True)

    img_root = '/home/hsy/research/car_lane/TT100K/data/xunlian/images/'
    length = len(imgs.keys())
    for idx, img_name in enumerate(imgs.keys()):
        img_info = imgs[img_name]
        img_path = img_info['path'].replace('test', 'val')

        if 'other' in img_path:
            continue
        img = cv2.imread(img_root+img_path)
        height, width = img.shape[:2]
        lines = []
        if len(img_info['objects']) > 0:
            for target in img_info['objects']:
                category = target['category']
                cls = labels.index(category)
                if cls == -1:
                    print(img_info)
                bbox = target['bbox']
                xmin, ymin, xmax, ymax = float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
                w, h = (xmax-xmin)/width, (ymax-ymin)/height
                cx, cy = ((xmin+xmax)/2)/width, ((ymin+ymax)/2)/height
                line = '%s %.8f %.8f %.8f %.8f\n' % (cls, cx, cy, w, h)
                lines.append(line)

        label_root = train_label_root if 'train' in img_path else test_label_root
        with open(label_root+'%s.txt' % img_name, 'w') as f:
            f.writelines(lines)

        print('\r %d of %d has finished' % (idx+1, length), end='')


# 移动TT100K
def t12():
    root = '/home/hsy/research/car_lane/TT100K/data/xunlian/images/'
    dst = '/home/hsy/research/car_lane/carTail_TT100K/images/'

    for split in ['train', 'val']:
        img_root = root+split
        dst_root = dst+split

        for idx, img_path in enumerate(glob.glob(img_root+'/*.*')):
            shutil.copy(src=img_path, dst=dst_root)
            shutil.copy(src=img_path.replace('images', 'labels').replace('.jpg', '.txt'), dst=dst_root.replace('images', 'labels'))

            print('\r%d of %d has finished' % (idx+1, len(glob.glob(img_root+'/*.*'))), end='')


if __name__ == '__main__':
    print()
    # t2()
    t3()
    # t4()
    # t6()
    # t7()
    # t8()
    # t9()
    # t10(True)
    # t11()
    # t12()
    # print(torch.distributed.is_available())




