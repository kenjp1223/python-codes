# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:19:03 2019

@author: Stuber_Lab
"""

#Create a high pass filter

import numpy as np
from scipy import ndimage
import cv2
import os 
import matplotlib.pyplot as plt
import skimage.filters as filt

def ecdf(DATA):
    """Compute ECDF for a one-dimensional array of measurements."""
    data = DATA.copy()
    data = data[data>0]
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1., n+1) / n
    
    return x, y

def distance(point,coef):
    return abs((coef[0]*point[0])-point[1]+coef[1])/math.sqrt((coef[0]*coef[0])+1)

def distance_point(p1,p2,p3):
    d= np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)) #p1,p2 start and end of line, p3 point
    return d

def get_rad(p1,p2):
    radian = math.atan2(p2[1] - p1[1],p2[0] - p1[0])
    return radian

#get the index of the peak of an array from a line. line defined by p1, p2
def get_rot_peak(array,p1,p2):
    rad = get_rad(p1,p2)
    #get rotation matrix
    c, s = np.cos(rad), np.sin(rad)
    R = np.array(((c,-s), (s, c)))
    
    #rotate the array
    X = np.dot(array,R)
    #get the index from peak
    ind = np.where(X[:,1] == np.max(X[:,1]))[0][0]
    return ind

def get_thresh_ecdf(data,p=1):
    x,y = ecdf(data)
    array = np.dstack([x,y])[0]
    
    p1 = array[0]
    p2 = array[int((len(array)-1)*p)]# you can avoid the outliers by using p% of the data
    ind = get_rot_peak(array,p1,p2)
    thresh = x[ind]
    #for debug
    '''
    plt.plot(x,y,'.')
    plt.axvline(x=thresh,ls = ':')
    plt.axvspan(xmin = thresh*0.75,xmax=thresh*1.25,color = 'cyan',alpha = 0.5)
    plt.show()
    plt.close()
    '''
    return thresh

DIR = r'\\172.25.144.34\Apotome\Ken\2019\190930 PVN Esr2-mCherry m92\m92\converted\r_filter'
raw_dir = r'\\172.25.144.34\Apotome\Ken\2019\190930 PVN Esr2-mCherry m92\m92\converted\r'
img_list = os.listdir(raw_dir)

for IMG in img_list:
    img_name = IMG.replace('.tif','')
    img = cv2.imread(raw_dir +'\\' + img_name + '.tif',-1)
    #check the shape of img
    #print(np.shape(img))
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    # Load the data...
    data = np.array(img, dtype=float)
    
    # Another way of making a highpass filter is to simply subtract a lowpass
    # filtered image from the original. Here, we'll use a simple gaussian filter
    # to "blur" (i.e. a lowpass filter) the original.
    sigma = 3
    lowpass = ndimage.gaussian_filter(data, sigma)
    gauss_highpass = data - lowpass
    gauss_highpass[gauss_highpass <0] = 0
    gauss_highpass = gauss_highpass.astype('uint16')
    #cv2.imwrite(DIR +'\\'+img_name+'_highpass_gauss_sigma' + str(sigma) + '.tif',gauss_highpass)
    
    #identify the background signal intensity
    #use yen thresholding algorithm, this was checked by a number of sample images
    thresh = filt.threshold_yen(gauss_highpass)
    print('Threshold : ' + str(thresh))
    img_cutoff = gauss_highpass.copy()
    #cut off anything below background
    img_cutoff[gauss_highpass < thresh] = 0
    img_cutoff = img_cutoff.astype('uint16')
    cv2.imwrite(DIR +'\\'+img_name+'.tif',img_cutoff)

