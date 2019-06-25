import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import matplotlib.pyplot as plt

def cuting(image_path, save_dir, size=128):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    print(image.shape,image_name)
 #   plt.imshow(image)
  #  plt.show()
    h, w = image.shape[0], image.shape[1]
    a = w // size + 1 # 列
    b = h // size + 1 # 行
    image_tmp = np.zeros((b*size, a*size, 3), dtype=np.uint8)
    image_tmp[:h, :w, :] = image[:, :, :]
    for i in range(b):
        for j in range(a):
            index = a*i+j+1
            cut_image_path = os.path.join(save_dir, image_name.split('.')[0]+'_'+str(index)+'.'+image_name.split('.')[1])
            cut_image = image_tmp[i*size:(i+1)*size, j*size:(j+1)*size]
            cv2.imwrite(cut_image_path, cut_image)
if __name__ == '__main__':
    #img_path = glob.glob('/media/lab/data/2018枝江市样方数据/22/*.tif')
    #for imgl in img_path:
    cuting('E:/aws_data/3_paris/AOI_3_Paris_Train/AOI_3_Paris_Train/t/MUL-PanSharpen_AOI_3_Paris_img28.tif','E:/aws_data/3_paris/AOI_3_Paris_Train/AOI_3_Paris_Train/t')




