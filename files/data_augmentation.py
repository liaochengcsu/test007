from PIL import Image
import sys
import cv2
import glob
import os


def fill_image(image):
    width, height = image.size
    print(width, height)

    new_image_length = width if width > height else height

    print(new_image_length)

    # new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='black')

    if width > height:

        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image


def cut_image(image):
    width, height = image.size
    item_width = int(width / 2)
    item_height = int(height/2)
    box_list = []
    count = 0
    for j in range(0, 2):
        for i in range(0, 2):
            count += 1
            box = (i * item_width, j * item_height, (i + 1) * item_width, (j + 1) * item_height)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('/media/lab/data/SceneChangeDet/OSCDdata2/gt/_' + str(index) + '.png')
        index += 1


if __name__ == '__main__':
    img_path = glob.glob('/media/lab/data/SceneChangeDet/OSCDdata2/t1/*.png')
    for imgl in img_path:
        img = Image.open(imgl)
        img_name = os.path.basename(imgl).split('.')[0]
    #file_path = "/media/lab/data/SceneChangeDet/OSCD/63_dataset/Onera Satellite Change Detection dataset - Images/aguasclaras/pair/img1.png"
    # 打开图像
    #image = Image.open(file_path)
    # 将图像转为正方形，不够的地方补充为白色底色
        #image = fill_image(img)
        # 分为图像
        image_list = cut_image(img)
        # 保存图像
        #save_images(image_list)
        index = 1
        for image in image_list:
            image.save('/media/lab/data/SceneChangeDet/OSCDdata3/t1/'+ img_name +'_' + str(index) + '.png')
            index += 1
