



"""
读取测试集txt，然后对图像进行检测，并保存检测结果
"""

import os
from PIL import Image
from yolo import YOLO

if __name__ == '__main__':

    yolo = YOLO()


    crop = False
    count = False


    detect_save = True
    detect_save_path = "detection_res/predict_imgs/"


    VOCdevkit_path = 'E:/Data-Center/Datatset/Object_Dection/ExDarkDatasetVOC'



    image_id = open(os.path.join(VOCdevkit_path, "ImageSets/test.txt")).read().strip().split()
    print(f"测试集数量为：{len(image_id)}")


    for img in image_id:
        img_path = os.path.join(VOCdevkit_path, "JPEGImages", img)
        inference_img = Image.open(img_path)


        r_image = yolo.detect_image(inference_img, crop=crop)



        if detect_save:
            if not os.path.exists(detect_save_path):
                os.makedirs(detect_save_path)


            img = img.lower()
            img = os.path.splitext(img)[0]
            r_image.save(os.path.join(detect_save_path, img + ".png"), quality=95, subsampling=0)
