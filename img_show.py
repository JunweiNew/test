import cv2
import os
import numpy as np
import argparse

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
img_list=[]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--no-sort', action='store_true')
    opts = parser.parse_args()

    for root,dirs,files in os.walk(opts.image_dir):
        for file in files:
            if file.split('.')[1] in ['png','jpg','bmp','tif']:
                img_path=os.path.join(opts.image_dir,file)
                img_list.append(img_path)


    for img_path in img_list:
        img_name=os.path.basename(img_path)
        cv2.namedWindow(img_name)
        src_img = cv2.imread(img_path)
        img = src_img

        while True:
            cv2.imshow(img_name, img)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
