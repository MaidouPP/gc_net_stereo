**gc_net_stereo** is the implementation according to paper: **End-to-End Learning of Geometry and Context for Deep Stereo Regression** which can be found in https://arxiv.org/abs/1703.04309 



1. how to run:

   CUDA_VISIBLE_DEVICES=n python main.py --gpu n --phase train --max_steps 50000 --learning_rate 0.00005 --output_dir /home/users/shixin.li/segment/gc-net/log/0508 --pretrain true

2. Before you run the code on a dataset, you need to generate an image list file, where image file paths are stored. You can use gen_image_list.py to generate it.

3. Scene Flow dataset uses pfm format, you probably want to use readPFM.py to read.