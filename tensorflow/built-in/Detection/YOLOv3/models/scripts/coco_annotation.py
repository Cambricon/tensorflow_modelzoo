import os
import argparse
import xml.etree.ElementTree as ET

def convert_coco_annotation(data_root, data_type, save_anno_path, label_root, use_difficult_bbox=True):

    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    img_inds_file = os.path.join(label_root, data_type + "_inds.txt")
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(save_anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_root, data_type, image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(label_root, data_type, image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                #difficult = obj.find('difficult').text.strip()
                #if (not use_difficult_bbox) and(int(difficult) == 1):
                #    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            # print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./path_to/datasets/COCO17/")
    parser.add_argument("--label_path", default="data/dataset/coco17_pascal_format/")
    parser.add_argument("--train_annotation", default="./data/dataset/coco17_train.txt")
    parser.add_argument("--val_annotation",  default="./data/dataset/coco17_val.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.val_annotation):os.remove(flags.val_annotation)


    num1 = convert_coco_annotation(data_root=flags.data_path,
                                data_type='train2017',
                                label_root=flags.label_path,
                                save_anno_path=flags.train_annotation,
                                use_difficult_bbox=False)

    num2 = convert_coco_annotation(data_root=flags.data_path,
                                data_type='val2017',
                                label_root=flags.label_path,
                                save_anno_path=flags.val_annotation,
                                use_difficult_bbox=False)

    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1 + num2, num3))


