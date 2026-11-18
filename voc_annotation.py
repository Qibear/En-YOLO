"""
本文件将使用官方的数据集划分进行训练

在使用官方的训练数据集的时候需要注意，txt文件的每行内容都带有后缀
"""

import os
import random
import xml.etree.ElementTree as ET
import numpy as np

from utils.utils import get_classes

annotation_mode = 2
classes_path = 'model_data/ex_dark.txt'


VOCdevkit_path = '/root/autodl-tmp/exdark'





trainval_percent = 0.9
train_percent = 0.9
VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(image_name, txt_name):
    """
    本函数将根据图像名称读取对应的xml文件，然后对其进行转换，并将转换结果存储在txt文件中
    注意这里的image_name其实是带图像后缀的（在读取xml文件的时候是没有影响的，但是在读取图像的时候需要注意）
    """
    in_file = open(os.path.join(VOCdevkit_path, f'Annotations/{image_name}.xml'), encoding='utf-8')

    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        txt_name.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


def printTable(List1, List2):
    for m in range(len(List1[0])):
        print("|", end=' ')
        for n in range(len(List1)):
            print(List1[n][m].rjust(int(List2[n])), end=' ')
            print("|", end=' ')
        print()


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集路径错误")


    if annotation_mode == 0 or annotation_mode == 2:

        print("生成训练 txt 文件")
        type_index = 0
        for year, image_set in VOCdevkit_sets:



            image_ids = open(os.path.join(VOCdevkit_path, f'ImageSets/{image_set}.txt'),
                             encoding='utf-8').read().strip().split()


            txt_file = open(f'{year}_{image_set}.txt', 'w', encoding='utf-8')
            en_txt_file = open(f'en_{year}_{image_set}.txt', 'w', encoding='utf-8')



            for image_id in image_ids:
                dir_path = os.path.abspath(VOCdevkit_path)


                txt_file.write(f'{dir_path}/JPEGImages/{image_id}')
                en_txt_file.write(f'{dir_path}/Enhanced_JPEG/{image_id}')


                convert_annotation(image_id, txt_file)
                convert_annotation(image_id, en_txt_file)

                txt_file.write('\n')
                en_txt_file.write('\n')

            photo_nums[type_index] = len(image_ids)
            type_index += 1
            txt_file.close()

        print("train.txt、val.txt 生成完毕")


        str_nums = [str(int(x)) for x in nums]
        tableData = [classes, str_nums]
        colWidths = [0] * len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集太小，请增大 epoch")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请检查classes_path是否准确、检查标签是否正确")
