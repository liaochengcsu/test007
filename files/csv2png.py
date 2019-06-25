import os
import numpy as np
import pandas as pd
import skimage.io
import skimage.draw
import matplotlib.pyplot as plt



# 存放标签的文件夹
label_dir = "./label"
if not os.path.exists(label_dir):
    os.mkdir(label_dir)

# csv文件路径
csv_path = "Solutions.csv"
df = pd.read_csv(csv_path)

image_ids = pd.unique(df.ImageId)
total_num = len(image_ids)
for i, image_id in enumerate(image_ids):
    print("进度：%d%%  生成第%d张图的标签" %(int((i+1)/total_num*100), (i+1)))

    image_annotation = df[df.ImageId == image_id]
    mask = np.zeros([650, 650], dtype=np.uint8)
    for _, row in image_annotation.iterrows():
        # 获取实例对应的id
        BuildingId = row['BuildingId']
        pix = row['PolygonWKT_Pix'].lstrip("POLYGON ")
        if pix != "EMPTY":
            # 获取每个实例对应的坐标列表
            pix = pix.strip("()")
            coord_list = pix.split(",")
            # 将实例的坐标列表分解成x和y两个列表
            x_list = []
            y_list = []
            for coord in coord_list:
                coord = coord.split(" ")
                x = round(float(coord[0].lstrip("(")))
                y = round(float(coord[1].lstrip("(")))
                x_list.append(x)
                y_list.append(y)
            # 将坐标列表转换为polygon
            yy, xx = skimage.draw.polygon(y_list, x_list)
            # 生成mask
            mask[yy, xx] = 255

    label_path = os.path.join(label_dir, image_id+".png")
    skimage.io.imsave(label_path, mask)
