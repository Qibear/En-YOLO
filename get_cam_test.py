



"""
获取测试集的完成cam图
"""

from PIL import Image
import os
from yolo import YOLO
from tqdm import tqdm

yolo = YOLO()

data_dir = r"E:Data-Center\Datatset\Object_Dection\ExDark\ExDarkDatasetVOC"

txt_file_path = os.path.join(data_dir, "ImageSets", "test.txt")
image_folder_path = os.path.join(data_dir, "JPEGImages")


with open(txt_file_path, 'r') as txt_file:
    lines = txt_file.readlines()


for line in tqdm(lines):
    image_path = line.strip()
    image_full_path = os.path.join(image_folder_path, image_path)
    res_save = "./cam_out_best/" + image_path


    if os.path.exists(image_full_path):
        img = Image.open(image_full_path)


        yolo.detect_heatmap(img, res_save)
