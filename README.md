# Mask-RCNN-annotation
This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.<br />
Refer to the most popular `keras` implemention here :https://github.com/matterport/Mask_RCNN

# Geting started
I only accomplish the annotation of the inference part and turn to another work which seems better.<br/>
Thus merely evaluation introduction provided.  <br/>
* **install**<br/>
No cuda files included ,the  project is friendly enough to greenhand. <br/>
```python setup.py develop```<br/>
Develop is strongly recommended!!
* **run demo.py for detection**<br/>
I rewrite the demo example for more custom realization.<br/>
(and some code are different from the source in `visualize.py`)<br/>
You only need to adjust the root directory in `demo.py` file , and run:<br/>
```python demo.py```<br/>
btw , I believe the annotation can not be more elaborate for your operation , you can modify what is required by yourself.<br/>

# Detection result
(Images are showed through matplotlib ,the blank around pictures really trouble me ,and cause my abandon on the code ,the probelem is left  :p )
![](https://github.com/ming71/Mask-RCNN-annotation/blob/master/output/0.7848402349160818.jpg)
![](https://github.com/ming71/Mask-RCNN-annotation/blob/master/output/0.3278841376282916.jpg)
