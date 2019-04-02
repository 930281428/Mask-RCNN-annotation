import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import ipdb
from keras.utils import plot_model

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#路径设置
def path_set():
    global ROOT_DIR
    global MODEL_DIR
    global COCO_MODEL_PATH
    global IMAGE_DIR

    ROOT_DIR = os.path.abspath("/py/Mask R-CNN/Mask_RCNN-tf")             # 当前根目录
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")                            # 存放训练日志和模型文件路径
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5") # 加载训练模型的路径
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")                          # 待检测图片的文件夹路径

def set_config():
    global config 
    global class_names

    class InferenceConfig(coco.CocoConfig):
        #该类继承自CocoConfig，而后者又继承自Config，为配置文件的基类
        GPU_COUNT = 1           # Set batch size to 1 for inference 
        IMAGES_PER_GPU = 1      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    config = InferenceConfig()
    # config.display()            #打印配置信息
    
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

   

# 整个文件夹下的依次处理显示
def detect_on_folder(save_flag):
    file_names = next(os.walk(IMAGE_DIR))[2]    #返回含有所有文件名的列表
    for i in range(len(file_names)):
        #skimage.io.imread 读取图片的方返回方式是 [h w c]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
        results = model.detect([image], verbose=0)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                  class_names, r['scores'],save_flag=save_flag)

# 单张图片的处理显示：
# mode: 'random' 自动在image文件夹下选一张处理
# mode: 'path'   处理给定路径的图片
# save_flag默认False,改动了原作者的可视化程序，可以保存图片
def detect_on_single(mode,save_flag):
    if mode is 'random':
        file_names = next(os.walk(IMAGE_DIR))[2]    #返回含有所有文件名的列表
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    else:
        image = skimage.io.imread(mode)
    results = model.detect([image], verbose=1)  #注意传入的image须为list
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                              class_names, r['scores'],save_flag=save_flag)

#将模型保存为PNG图片输出
def save_model_png():
    plot_model(model.keras_model, show_shapes=True,to_file='model.png')


if __name__ == '__main__':
    path_set()  #在这个函数下设置相关目录
    set_config()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    # plot_model(model.keras_model, show_shapes=True,to_file='model.png')

    # detect_on_single('/py/Mask R-CNN/Mask_RCNN-tf/images/1.jpg',save_flag=True) 
     detect_on_folder(save_flag=True)

#    detect_on_single('/py/pic/hourse.jpg',save_flag=False)
