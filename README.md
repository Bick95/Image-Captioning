# Image-Captioning
Repo contains system for Training and Evaluating Image Captioning System based on Deep Learning.

To make things more readily comparable, we present some of our best results obtained with attention based and non-attention image captioning models.

#### Soft Attention-Based result 1:

![](/results/sa_result_1.png)

Caption: <start> a couple is shoveling snow wearing backpacks are walking together through the forest <end>
  
  
#### Soft Attention-Based result 2:

![](/results/sa_result_2.png)

Caption: <start> a brown and brown and white dog jumping into the ocean <end>
  
  
#### Soft Attention-Based result 3:

![](/results/sa_result_3.png)

Caption: <start> man riding a trampoline outside at a bench with a speech on a colored slide <end>

#### Non-Attention-Based result: 

![](/results/na_result.jpg)  


Real caption (Human made): <start> a boy in a white t-shirt is playing on a swing that is attached to a tree <end>
  
Our model's caption: <start> a young man with red and pink hair and the other on an exercise in a grassy field <end>

## Navigation through the Repository
### Master branch
The master branch contains an implementation of the image captioning system which has been adapted from a [Tensorflow Tutorial page on Image Captioning](https://www.tensorflow.org/tutorials/text/image_captioning) and which is based on the main principles presented in the research paper [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention by Xu et al. (2015)](https://arxiv.org/abs/1502.03044). 
### HardAttention2 branch
The branch [HardAttention2](https://github.com/Bick95/Image-Captioning/tree/HardAttention2) contains a more customizable implementation of the code on the master branch, which allows for more freedom in parameter choices and which is closer related to the original implementation presented in the aforementioned research paper. However, there seems to still be some bug present in this implementation, since learning does not seem to take place. Also, it contains an implementation of the Hard Attention module proposed in the aforementioned research paper. 
### MergerMasterHardAtt branch
The brach [MergerMasterHardAtt](https://github.com/Bick95/Image-Captioning/branches) is a hybrid between the former two branches, being closer related to the master branch implementation, but contains an implementation of the Hard Attention module as well. 
