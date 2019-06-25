import scipy
import pylab
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import PIL.Image as Image

Pad_D = 120 #to be filled on the down side
Pad_R = 120 #to be filled on the right side
Crop_Size = 512#cliped image patch size
Image_Size=5000#source image size
C_R_N = 10#number of patches on cow/row

img_dir = r"F:\Seg_Data\images"#path of the image to be clip
save_dar = r"F:\test_result_temp"#path to save the cliped image patch
crop_path = r"F:\Seg_Data\crop"#path to the image to be croped
index = 1000000

#clip the image to patches
# file_name = os.listdir(img_dir)
# for i in range(len(file_name)):
#     file = os.path.join(img_dir, file_name[i])
#     image = Image.open(file)
#     image = np.pad(image, ((0, Pad_D), (0, Pad_R), (0, 0)),'constant')  # 上下，左右，通道
#     image = Image.fromarray(image, mode='RGB')
#     for i in range(C_R_N):
#         for j in range(C_R_N):
#             roi_area = image.crop(
#                 [i * Crop_Size, j * Crop_Size, (i + 1) * Crop_Size, (j + 1) * Crop_Size, ])
#             out_name = os.path.join(save_dar, str(index) + ".png")
#             index = index + 1
#             roi_area.save(out_name)


#crop the patches to an Image as sources size

file_name = os.listdir(img_dir)
num=0
for n in range(len(file_name)):
    toImage = Image.new("L",(Image_Size+Crop_Size,Image_Size+Crop_Size))
    for y in range(C_R_N):
        # num = num + y * C_R_N
        for x in range(C_R_N):

            name = str(index + num) + '.png'
            fname = os.path.join(save_dar, name)
            fromImage = Image.open(fname)
            toImage.paste(fromImage, (y * Crop_Size, x * Crop_Size))
            num = num + 1
    roi_area = toImage.crop([0,0,Image_Size,Image_Size ])
    roi_area.save(os.path.join(crop_path,file_name[n]))

