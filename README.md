# Implementation for paper "Layered Embeddings for Amodal Instance Segmentation" 

**Note: the email associated with the paper is no longer valid. Please contact me at `yanfengliux@gmail.com`**

**An extension of this work into multi-class multi-object tracking can be found in [my thesis](https://digitalcommons.unl.edu/elecengtheses/111/)**

Paper link: [Springer](https://link.springer.com/chapter/10.1007/978-3-030-27202-9_9), [Arxiv](https://arxiv.org/abs/2002.06264)

![](https://i.imgur.com/q2y3VVN.png)

Fig 1. Image and training ground truth: front class mask, occlusion class mask, front instance mask, occlusion instance mask

![](https://i.imgur.com/N6eibon.png)

Fig 2. Network architecture

![](https://i.imgur.com/7u4QY0N.png)

Fig 3. Failure case where there is a three-stack

![](https://i.imgur.com/lf9SKEF.png)

Fig 4. Incomplete instance mask for the object at the bottom in the failure case

![](https://i.imgur.com/WtB3uCr.png)

Fig 5. Mask R-CNN fails when there many objects of the same class and the bounding boxes are too crowded
