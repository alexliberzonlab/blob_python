# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:36:18 2016

@author: User
"""

from Blob_Reader import blobReader, extractor
import os
import matplotlib.pyplot as plt
from skimage.io import imread , imshow


directory = 'D:/Liran_2018_July/160718'
os.chdir(directory)



blobfiles = ['b0.dat',
             'b1.dat',
             'b2.dat',
             'b3.dat']



FrameStart = None    # Leave None to go over all the frames
FrameEnd = None    # Leave None to go over all the frames

# parameters for the fitness coefficient:
#window = 5  
#N = 1
#fit_range = [0.5,1.5]


ex = extractor(directory , blobfiles)
ex.load(FrameStart ,FrameEnd)
#ex.search_frames(window,  N, fit_range)



#ex.plot_frame_count()
#ex.plot_frames(FrameStart,FrameEnd) 
# ex.gen_Target_Files()
#plt.hold(True)
#imshow(imread('cam0.tif'))