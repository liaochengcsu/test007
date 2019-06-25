import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def convert(src_dir, dst_dir, bands=(0, 1, 2)):
    """
    :param src_dir: 原始图片所在的目录
    :param dst_dir: 转化后保存的目录
    :param bands: R、G、B三个波段对应的通道编号
    :return:
    """

    if not os.path.exists(dst_dir):
        raise "%s is not exxst!" % src_dir
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    file_list = os.listdir(src_dir)
    for name in file_list:
        src_path = os.path.join(src_dir, name)
        dis_name=os.path.splitext(name)[0] + '.png'
        dst_path = os.path.join(dst_dir, dis_name)


        src_img = tifffile.imread(src_path)
        height, width = src_img.shape[:2]
        print(src_img.shape[:3])
        dst_img = np.zeros((height, width, 3), dtype=np.uint8)

        for i, band in enumerate(bands):
            try:
                max = np.max(src_img[:, :, band])
                min = np.min(src_img[:, :, band])
                dst_img[:, :, i] = np.array((src_img[:, :, band] - min) / (max - min) * 255, dtype=np.uint8)
            except:
                dst_img[:, :, i] = np.array(src_img[:, :, band] / 65535 * 255, dtype=np.uint8)

        tifffile.imsave(dst_path, dst_img)


if __name__ == '__main__':
    src_dir = r"D:\1\data_test\image"
    dst_dir = r"D:\1\data_test\result"
    convert(src_dir, dst_dir, bands=(2, 1, 0))

