# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:26:49 2018

@author: lhf
"""
import os,time,random
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse

from mpl_toolkits.axes_grid1 import host_subplot
from data import load_batch,prepare_data
from model import hrnet
#
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Number of images in each batch')
#
# parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
# parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
# parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
# parser.add_argument('--clip_size', type=int, default=512, help='Width of cropped input image to network')
# parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs to train for')
# parser.add_argument('--h_flip', type=bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
# parser.add_argument('--v_flip', type=bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
# parser.add_argument('--rotation', type=bool, default=False, help='randomly rotate, the imagemax rotation angle in degrees.')
# parser.add_argument('--num_val_images', type=int, default=800, help='Number of image to valid for')
# parser.add_argument('--valid_step', type=int, default=10, help="Number of step to validation")

#
parser.add_argument('--checkpoint_step', type=int, default=20, help='How often to save checkpoints (epochs)')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--clip_size', type=int, default=480, help='Width of cropped input image to network')
parser.add_argument('--num_epochs', type=int, default=140, help='Number of epochs to train for')
parser.add_argument('--h_flip', type=bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--color', type=bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=bool, default=True, help='randomly rotate, the imagemax rotation angle in degrees.')
parser.add_argument('--num_val_images', type=int, default=800, help='Number of image to valid for')
parser.add_argument('--valid_step', type=int, default=10, help="Number of step to validation")


args = parser.parse_args()


img=tf.placeholder(tf.float32,[args.batch_size,args.crop_height,args.crop_width,3])
label=tf.placeholder(tf.float32,[args.batch_size,args.crop_height,args.crop_height,1])
pred=hrnet(img)
#pred=tf.nn.sigmoid(pred)

# def focal_loss(pred,labels,alpha,gamma):
#     zeros=tf.zeros_like(pred,dtype=pred.dtype)
#     pos_corr=tf.where(labels>zeros,labels-pred,zeros)
#     neg_corr=tf.where(labels>zeros,zeros,pred)
#     fl_loss=-alpha*(pos_corr**gamma)*tf.log(pred)-(1-alpha)*(neg_corr**gamma)*tf.log(1.0-pred)
#     return tf.reduce_sum(fl_loss)

#pred_test = pspnet(img, is_training=False)
#pred_test = tf.nn.sigmoid(pred_test)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # sigmoid_cross_entropy_loss = tf.reduce_mean(focal_loss(pred=pred,labels=label,alpha=0.25,gamma=2))
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)
    sigmoid_cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred))
    sigmoid_cross_entropy_loss=sigmoid_cross_entropy_loss
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)
train_img, train_label = prepare_data()



num_batches=len(train_img)//args.batch_size
saver=tf.train.Saver(var_list=tf.global_variables())




def load():
    import re
    print("Reading checkpoints...")
    checkpoint_dir = './checkpoint/'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print("Checkpoint {} read Successed".format(ckpt_name))
        return True, counter
    else:
        print("Checkpoint not find ")
        return False, 0

def train():

    tf.global_variables_initializer().run()

    could_load, checkpoint_counter = load()
    if could_load:
        start_epoch = (int)(checkpoint_counter / num_batches)
        start_batch_id = checkpoint_counter - start_epoch * num_batches
        counter = checkpoint_counter
        print("Checkpoint Load Successed")

    else:
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        print("train from scratch...")

    train_iter=[]
    train_loss=[]

    utils.count_params()
    print("Total image:{}".format(len(train_img)))
    print("Total epoch:{}".format(args.num_epochs))
    print("Batch size:{}".format(args.batch_size))
    print("Learning rate:{}".format(args.learning_rate))
    print("Checkpoint step:{}".format(args.checkpoint_step))

    print("Data Argument:")
    print("h_flip: {}".format(args.h_flip))
    print("v_flip: {}".format(args.v_flip))
    print("rotate: {}".format(args.rotation))
    print("clip size: {}".format(args.clip_size))


    for i in range(start_epoch,args.num_epochs):
        id_list = np.random.permutation(len(train_img))

        epoch_time=time.time()
        for j in range(start_batch_id,num_batches):
            img_d=[]
            lab_d=[]
            for ind in range(args.batch_size):
                id = id_list[j * args.batch_size + ind]
                img_d.append(train_img[id])
                lab_d.append(train_label[id])

            x_batch, y_batch = load_batch(img_d,lab_d,args)
            feed_dict = {img: x_batch,
                         label: y_batch

                         }
            loss_tmp = []
            _, loss, pred1 = sess.run([train_step, sigmoid_cross_entropy_loss, pred], feed_dict=feed_dict)
            loss_tmp.append(loss)
            if (counter % 100 == 0):
                tmp = np.mean(loss_tmp)
                train_iter.append(counter)
                train_loss.append(tmp)
                print('Epoch', i, '|Iter', counter, '|Loss', tmp)
            counter += 1
        start_batch_id=0
        print('Time:', time.time() - epoch_time)

        #if((i+1)%10==0 ):#lr dst from 10 every 10 epoches by 0.1
            #learning_rate = 0.1 * learning_rate
        #last_checkpoint_name = "checkpoint/latest_model_epoch_" + "_pspet.ckpt"
        # print("Saving latest checkpoint")
        # saver.save(sess, last_checkpoint_name)


        if((i+1)%args.checkpoint_step==0):#save 20,30,40,50 checkpoint
            args.learning_rate=0.1*args.learning_rate
            print(args.learning_rate)

            saver.save(sess,'./checkpoint/model.ckpt',global_step=counter,write_meta_graph=True)


            """
            host = host_subplot(111)
            plt.subplots_adjust(right=0.8)
            p1, = host.plot(train_iter, train_loss, label="training loss")
            host.legend(loc=5)
            host.axis["left"].label.set_color(p1.get_color())
            host.set_xlim([0, counter])
            plt.draw()
            plt.show()
            """
            fig1, ax1 = plt.subplots(figsize=(11, 8))
            ax1.plot(train_iter, train_loss)
            ax1.set_title("Training loss vs Iter")
            ax1.set_xlabel("Iter")
            ax1.set_ylabel("Training loss")
            plt.savefig('Training loss_vs_Iter.png')
            plt.clf()

        remain_time=(args.num_epochs - 1 - i) * (time.time() - epoch_time)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        print("Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s))
    #saver.save(sess, './checkpoint/model.ckpt', global_step=counter)




with tf.Session() as sess:
    train()

