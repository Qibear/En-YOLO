



"""
读取图像，然后将显示热力图
"""

from PIL import Image
from yolo import YOLO

yolo = YOLO()

img_path = "img/2015_00058.jpg"
res_save = 'model_data/heatmap_vision.png'

image = Image.open(img_path)
yolo.detect_heatmap(image, res_save)
