## Context Encoders: Feature Learning by Inpainting

This is the Pytorch implement of [CVPR 2016 paper on Context Encoders](https://arxiv.org/pdf/1604.07379.pdf)


### 1) Semantic Inpainting Demo

1. Install PyTorch http://pytorch.org/

2. Clone the repository
  ```Shell
  git clone https://github.com/Computer-Vision-IIITH-2021/project-jaathiratnalu
  ```
3. Demo
	
    Go to src2 direcotry

    Download pre-trained model(L2=0.9999) on Paris Streetview from
    [One Drive](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/nihar_potturu_students_iiit_ac_in/Es_vjOjmrAxLorFjaJ4jAf0BBRC_FVd_QifGgVly9HtzuQ?e=QdDdp1) 
    ```Shell
    Copy the downloaded model in the 'model' directory
    # Inpainting one image 
    python test_one.py --netG 'path to model file' --test_image 'path to test image'
    ```

### 2) Train on your own dataset
1. Build dataset

    Put your images under dataset/train,all images should under subdirectory

    dataset/train/subdirectory1/some_images
    
    dataset/train/subdirectory2/some_images

    ...
    
    **Note**:For Google Policy,Paris StreetView Dataset is not public data,for research using please contact with [pathak22](https://github.com/pathak22).
    You can also use [The Paris Dataset](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) to train your model

2. Train
```Shell
python train.py --cuda --wtl2 0.999 --niter 200
```

3. Test

    This step is similar to [Semantic Inpainting Demo](#1-semantic-inpainting-demo)

    
