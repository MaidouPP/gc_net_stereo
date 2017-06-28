import numpy as np
import re
import sys
from PIL import Image
from scipy import misc

def load_pfm(ad):
    file=open(ad)
    color = None
    width = None
    height = None
    header = file.readline().rstrip()
    endian = ''
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Wrong!!!')

    scale = float(file.readline().rstrip())
    if scale<0: 
        endien = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian+'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    img = Image.fromarray(data)
    img = img.rotate(180)
    file.close()
    return np.array(img)

if __name__ == '__main__':
    f = '/home/users/shixin.li/segment/data_stereo/left_gt/0001.pfm'
    img = Image.fromarray(load_pfm(f))
    print np.array(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save('test0001.png')
