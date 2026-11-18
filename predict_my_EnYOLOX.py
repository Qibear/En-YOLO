import os
from PIL import Image
from yolo import YOLO
from tqdm import tqdm

if __name__ == '__main__':

    yolo = YOLO()


    crop = False
    count = False



    detect_save = True
    detect_save_path = r"E:\EnYOLO_Test\04_EnYOLOX"



    VOCdevkit_path = 'E:/Data-Center/Datatset/EnYOLO_Dataset/ExDarkDatasetVOC'



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
            r_image.save(os.path.join(detect_save_path, img + ".jpg"), quality=95, subsampling=0)
