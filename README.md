**gc_net_stereo** is the implementation according to paper: **End-to-End Learning of Geometry and Context for Deep Stereo Regression** which can be found in https://arxiv.org/abs/1703.04309 



1. how to run:

   CUDA_VISIBLE_DEVICES=n python main.py --gpu n --phase train --max_steps 50000 --learning_rate 0.00005 --output_dir /home/users/shixin.li/segment/gc-net/log/0508 --pretrain true

   n为指定的gpu号，不是微调阶段的话置pretrain为false

2. 跑数据集之前生成一个image list，里面存的是图片的地址，也可以存成绝对的．例子见本文件夹下的train.lst，生成代码是gen_image_list.py

3. 因为Scene Flow数据集用的groundtruth是.pfm格式，所以用readPFM.py来读取．

4. 现在这个程序跑的是Scene Flow数据集，之前输入用灰度图，现在用的是RGB的图，收敛效果都不是很好．最新的跑了２w+次的checkpoint在0518文件夹下．

5. 这份代码不保证工作，已知收敛效果并不好。不再继续更新。
