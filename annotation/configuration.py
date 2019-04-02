1.自定义网络配置十分灵活，在mrcnn/config.py中十分详细地定义了网络配置的相关参数；
  自定义网络时只需要import该文件并继承Config类，然后修改相关的参数，即可实现自定义
  
2.注意：这些参数放在构造函数外部，因而在执行（不是调用）自定义类时，会从头执行一遍类内语句，完成对下面变量的赋值。

3.举个例子：
    demo.py中有如下继承：
        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
    而coco.CocoConfig类定义为：
        class CocoConfig(Config):
            NAME = "coco"
            IMAGES_PER_GPU = 2
            NUM_CLASSES = 1 + 80  # COCO 80 classes
    最终还是继承的父类Config，但是继承过程中针对自己需要实例化的特点，改变了部分参数（如inference的image_pre_gpu只需要为1）
    
4.Config类中除了构造函数外，还有一个display方法，可以打印config配置信息

5.下面说明部分参数的含义：（作者说的非常清楚简直是完美，建议直接看原程序annotation）
    NAME = None  # 自己随便取
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  #越大越好，对于12G显存可以放1024*1024两张图片
    STEPS_PER_EPOCH = 1000  # Number of training steps per epoch不建议太小
    VALIDATION_STEPS = 50   #越大越好，但是收敛慢
    BACKBONE = "resnet101"  #目前支持的backbone只有这个和resnet50
    COMPUTE_BACKBONE_SHAPE = None
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]   #FPN层的降采样步长，这个是Resnet101的参数，改bacckbone需要调整
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024   #分类的全连接层尺寸
    TOP_DOWN_PYRAMID_SIZE = 256 #自底向上搭建特征金字塔的层数，即FPN通道数
    NUM_CLASSES = 1  # 包含背景在内的类别数（继承时自定义）
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) #正方形anchor的边长
    RPN_ANCHOR_RATIOS = [0.5, 1, 2] #anchor形状，1为正方形，0.5为扁长方形
    RPN_ANCHOR_STRIDE = 1       # Anchor stride正常为1就好了，如果设置2可以减少proposal的数目，降低运行负载和加速
    RPN_NMS_THRESHOLD = 0.7 #RPN产生proposal的NMS阈值
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256   #每张图片留下给RPN训练的anchor数目
    PRE_NMS_LIMIT = 6000    #nms之前选择的top-N的RoI数目
    POST_NMS_ROIS_TRAINING = 2000   #nms后保留的RoI数目(分别是训练和检测)
    POST_NMS_ROIS_INFERENCE = 1000
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200  #正样本的数目，作者用的512,但是实际很可能没有那么多正样本

    # Percent of positive ROIs used to train classifier/mask heads
    #正样本的bilin1：3
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]   
    MAX_GT_INSTANCES = 100  #设置使用单张图片最多的gt数目（最多只用100个gt训练）
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
    DETECTION_MAX_INSTANCES = 100   #最大的物体数
    DETECTION_MIN_CONFIDENCE = 0.7  #低于该值的RoI直接舍弃
    DETECTION_NMS_THRESHOLD = 0.3   #检测的NMS阈值
    LEARNING_RATE = 0.001   #都是优化参数，其中lr论文给的0.02,这里会梯度爆炸，可能是优化器工作不同导致
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001   
    LOSS_WEIGHTS = {        #不同损失的比重，便于更精确地调参
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    USE_RPN_ROIS = True #这个debug用的，一般都是True就行了
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    GRADIENT_CLIP_NORM = 5.0    # Gradient norm clipping梯度裁剪

