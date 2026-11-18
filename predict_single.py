import os
from PIL import Image
from yolo import YOLO

if __name__ == '__main__':

    img_path = "img/street.jpg"

    yolo = YOLO()
    crop = False
    count = False


    detect_save = False
    detect_save_path = "./detection_res/"

    inference_img = Image.open(img_path)


    r_image = yolo.detect_image(inference_img, crop=crop)
    r_image.show()


    if detect_save:
        if not os.path.exists(detect_save_path):
            os.makedirs(detect_save_path)


        img = img_path.lower()
        img = os.path.splitext(img)[0]
        r_image.save(os.path.join(detect_save_path, img + ".png"), quality=95, subsampling=0)
