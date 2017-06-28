import os
import re
import sys
import numpy as np
import random
import cv2
from scipy import misc
import glob
from PIL import Image
import h5py

NEGATIVE_LOW = 4
NEGATIVE_HIGH = 10
cal = 0

if __name__=='__main__':
    with h5py.File("/home/users/shixin.li/segment/Lecun_stereo_rebuild/trainPatches.hdf5", "w") as f1:
        with h5py.File("/home/users/shixin.li/segment/Lecun_stereo_rebuild/trainImgs.hdf5", "r") as f2:
            grpLeft = f1.create_group("left")
            grpRightPos = f1.create_group("rightPos")
            grpRightNeg = f1.create_group("rightNeg")
            while (cal!=50000):
                randomNumber = str(random.randint(0,193))
                imgL = Image.fromarray((f2['left/'+randomNumber][()]))
                imgR = Image.fromarray(f2['right/'+randomNumber][()])
                imgGround = f2['groundtruth/'+randomNumber][()]
 
                i = random.choice([m for m in xrange(370)])
                j = random.choice([m for m in xrange(1226)])
                o_positive = random.randint(-1,1)
                sign = random.choice([-1,1])
                o_negative = sign * random.randint(NEGATIVE_LOW, NEGATIVE_HIGH)

                if j-5>0 and i-5>0 and j+6<imgL.size[0] and i+6<imgL.size[1]:
                    imgLPatch = imgL.crop((j-5, i-5, j+6, i+6))
                    disp = int(float(imgGround[i][j])/256)
                    if np.sum(np.array(imgLPatch))==0 or disp==0:
                        continue
                    grpLeft.create_dataset(name=str(cal), data=imgLPatch)
                    if cal%10 is 0:
                      print "processing....", cal
                    if j+o_positive-disp+6 < imgL.size[0] and j+o_negative-disp+6 < imgL.size[0]:
                        imgRPatchPos = np.array(imgR.crop((j+o_positive-disp-5, i-5, j+o_positive-disp+6, i+6)))
                        grpRightPos.create_dataset(str(cal), data=imgRPatchPos)
                        imgRPatchNeg = imgR.crop((j+o_negative-disp-5, i-5, j+o_negative-disp+6, i+6))
                        grpRightNeg.create_dataset(str(cal), data=imgRPatchNeg)
                        cal+=1
