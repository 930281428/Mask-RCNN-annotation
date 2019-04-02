对网络结构的分析：https://zhuanlan.zhihu.com/p/40314107

本作者没有严格按照Keras的方法继承自基类Layer來定义自己的网络，而是完全自己定义（但是built，call等必要方法的函数名还是沿用了Layer的方法），这段网络定义近千行，下面逐个分析：

def __init__(self, mode, config, model_dir)
    构造函数，传入参数注意mode选择为['training', 'inference']中的一个字符即可;
    self.keras_model = self.build(mode=mode, config=config)搭建模型
    
def build(self, mode, config)
    1.Input: 搭建输入层 (注意：所有这里的输入定义都没有加batch，实际上后面都会有的，参加输出的图)
        (inference)输入分为三个：input_image（图像像素矩阵），input_image_meta（形状变换等信息），input_anchors
    2.ResNet backbone: 搭建自底向上的共享卷积层
        自定义callable的backbone（可以可以，还可以自己替换）
        resnet-50/101：
            C1：首先是一个padding和7*7卷积降采样+Maxpooling降采样，得到的作为C1
            C2：conv_block降采样，两个identity_block后输出C2
            C3：同C2
            C4：先conv_block降采样，然后根据不同backbone配置identity_block数目-->resnet-50:5个/resnet-50:22个
            C5：flag设置是否添加该层。如果添加同C2配置
    3.FPN: 搭建自顶向下的特征融合层：
        config.TOP_DOWN_PYRAMID_SIZE设置特征金字塔的通道数，所有用于融合的特征图经过卷积得到该通道数（同时上采样）再进行融合
        特征融合：
            P5:对C5卷积得到
            P4：将P5上采样后与C4融合（像素相加）得到
            P3：将P4上采样后与C3融合（像素相加）得到
            P2：将P3上采样后与C2融合（像素相加）得到
        卷积消除棋盘效应+P6生成：
            P2：对P2加3*3*TOP_DOWN_PYRAMID_SIZE（256）的same卷积得到
            P3：对P3加3*3*TOP_DOWN_PYRAMID_SIZE（256）的same卷积得到
            P4：对P4加3*3*TOP_DOWN_PYRAMID_SIZE（256）的same卷积得到
            P5：对P5加3*3*TOP_DOWN_PYRAMID_SIZE（256）的same卷积得到
            P6：P5 stride=2 Maxpooling降采样得到,用于RPN
        特征分类处理：
            rpn_feature_maps:    [P2,P3,P4,P5,P6]    送入RPN分类和回归，得到anchor的前景/背景鉴别得分,类别得分和定位修正信息；
            mrcnn_feature_maps:  [P2,P3,P4,P5]       后面进行ROI Align时的分割目标。
    4.Anchors
        对RPN五张特征图，生成：(256*256+128*128+64*64+32*32+16*16)*3=261888个anchor
        （很明显这个数目太大，所以RPN网络先计算每个anchor得分后，NMS之前提取top_k(如6k)个，NMS后只保留training 2000/inference 1000个）
    5.RPN输出proposals 
        rpn_model： 
            搭建每次处理一张特征图（遍历ft list即可处理6张）的rpn_model,输入为rpn_feature_maps list，输出经过concat得到三个输出
            输出：rpn_class_logits, rpn_class, rpn_bbox
            维度：[batch, num_anchors, 2] ，[batch, num_anchors, 2]，[batch, num_anchors, 4]
            功能：三者都参与loss计算；其中rpn_class和rpn_bbox还会用作后面的ROI层输入     
    6.RoI对proposals进行处理     (自定义ProposalLayer实现)
        网络前向输入：rpn_probs: [batch, num_anchors, 2]   (2=(bg prob, fg prob))
                    rpn_bbox:  [batch, num_anchors, 4]   (4=(dy, dx, log(dh), log(dw)))
                    anchors:   [batch, num_anchors, 4]   (4=(y1, x1, y2, x2))（归一化）
        输出：归一化的proposal   [batch, rois, 4]          (4=(y1, x1, y2, x2) , rois=1000/2000)
        具体流程：
            (1)提取scores（前景得分），deltas，anchors
            (2)根据scores筛一轮留下top_k=6000 个anchor(原来有26000+个太多了)(PRE_NMS_LIMIT=6000)
            -->此处后面大量使用utils.batch_slice进行提取
            (3)使用预测的偏移值deltas对anchor进行坐标调整（0-1边界限定）
            (4)NMS并填充得到固定数目的bbox,如[bs,1000,4]（POST_NMS_ROIS_TRAINING=2000,POST_NMS_ROIS_INFERENCE=1000）
     7.Network Heads
        detection heads (fpn_classifier_graph)
            输入特征：mrcnn_feature_maps，rpn_rois   [bs,1000,4](归一化)
            输出：   mrcnn_class_logits:       [batch, num_rois, NUM_CLASSES]   classifier logits (before softmax)
                    mrcnn_class:        [batch, num_rois, NUM_CLASSES]         classifier probabilities
                    mrcnn_bbox:  [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] 
            流程：
                (1)RoIAlign:输不同尺寸特征图，得到[batch,1000,7,7,256]输出
                (2)接两个7*7卷积实现的FC层，然后压缩维数1的维度，得到[bs,1000,1024]
                (3)Classifier head： 在(2)输出接80维Dense得到mrcnn_class_logits，再加一个softmax得到mrcnn_probs
                (4)BBox head： 在(2)输出接80*4维Dense,reshape成[bs,1000,81,4]得到mrcnn_bbox
        检测层(DetectionLayer)
            输入特征：[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
            输出：    检测结果：归一化的[batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            功能： 对输入的RoI根据分类回归结果进行bbox的修正计算，并且进行NMS，留下的box数目设置上限为DETECTION_MAX_INSTANCES=100
        mask heads (fpn_mask_graph)
            生成mask [bs,100,28,28,81]
        
        
        
        
        
        
        
            
    
   
   
   
