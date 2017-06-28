import numpy as np
import h5py
import os
import glob
from PIL import Image

def imageProcess():
    currPath="/home/users/shuo.cheng/segment/training/"
    os.chdir(currPath+"image_0/")
    nameL=glob.glob("*_10.png")
    os.chdir(currPath+"image_1/")
    nameR=glob.glob("*_10.png")
    os.chdir(currPath+"disp_noc/")
    namesG=glob.glob("*_10.png")
    
    with h5py.File("/home/users/shuo.cheng/segment/training/trainImgsKIT.hdf5", "w") as f:
        print "Normalizing and saving left images..."
        grp = f.create_group("left")
        for img in nameL:
            s = int(img.split('_')[0])
            if s<=3:
                trainImg = np.array(Image.open(currPath + 'image_0/' + img))
                trainImg = 2.0 * (trainImg - np.mean(trainImg)) / (trainImg.max() - trainImg.min())
                grp.create_dataset(name=str(s), data=trainImg)
        
        print "Normalizing and saving right imges..."
        grp = f.create_group("right")
        for img in nameR:
            s = int(img.split('_')[0])
            if s<=3:
                trainImg = np.array(Image.open(currPath + 'image_1/' + img))
                trainImg = 2.0 * (trainImg - np.mean(trainImg)) / (trainImg.max() - trainImg.min())
                grp.create_dataset(name=str(s), data=trainImg)
        
        print "Saving groundtruth..."
        grp = f.create_group("groundtruth")
        for img in namesG:
            s = int(img.split('_')[0])
            if s<=3:
                gt = np.array(Image.open(currPath + 'disp_noc/' + img))
                gt = gt / 256
                gt = gt.astype(int)
                grp.create_dataset(name=str(s), data=gt)

if __name__ == '__main__':
    imageProcess()
