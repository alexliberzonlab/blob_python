# -*- coding: utf-8 -*-
"""
This is a script for reading a blobRecorder flie.
written by Ron Shnapp
july-20-2016
"""

from struct import unpack
from pandas import DataFrame
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

             
#File2extrac = blobfiles[0]



class blobReader(object):
    '''
    a reader of blobrecorder generated files.
    the blobrecorder .dat files are Little-Endian binary files
    each file has 60 bytes of header
    each frame has 12 bytes of header
    blobs are 16 bytes long each
    
    - The blob data itself is stored in the attribute Blobs, which
    is a pandas dataframe object  
    
    - use ReadBlobFile('blob file Name') to read the data from the file 
    into the blobreader instance
    
    - use saveframes() to save the blob data on the disk as csv 
    '''
    def __init__(self):
        self.fname = ''
        self.HeaderSize = 15            # Each blobreader file descriptor is 15 bytes
        self.FrameHeaderSize = 3        # Each frame descriptor is 3 bytes
        self.BlobFeaturesSize = 16      # Each blob descriptor is 16 bytes
        self.FrameCount = 0             # number of frames of the object
        self.tStart = [0,0,0,0,0,0,0]   # recording start time
        self.tEnd = [0,0,0,0,0,0,0]     # recording end time
        # the blobs data itself as a pandas DataFrame object:
        self.Blobs = DataFrame(columns=['x0','x1','y0','y1','Xcog','Ycog','Area','time','Frame'])
        
        
    def __repr__(self):
        a = 'BlobReader object \n'
        b = 'Frames: %d \n'%self.FrameCount
        c = 'Record Start: %0.2d-%0.2d-%0.4d %0.2d:%0.2d:%0.2d.%0.2d \n'%tuple(self.tStart)
        d = 'Record End: %0.2d-%0.2d-%0.4d %0.2d:%0.2d:%0.2d.%0.2d \n'%tuple(self.tEnd)
        e = 'Blobs: %d \n'%len(self.Blobs)
        return '\n' + a + b + e + c + d
        

    def ReadSingleBlob(self , File, FrameHeader):
        '''
        will read the blob data from a blob file that was written
        in the "old style" without floating number coordinates
        '''
        x0 = (unpack('<h', File.read(2))[0]) # bounding box x0
        x1 = (unpack('<h', File.read(2))[0]) # bounding box x1
        y0 = (unpack('<h', File.read(2))[0]) # bounding box y0
        y1 = (unpack('<h', File.read(2))[0]) # bounding box y1
        A = (unpack('<i', File.read(4))[0])  # Area
        X = (unpack('<h', File.read(2))[0])  # x center of gravity
        Y = (unpack('<h', File.read(2))[0])  # y center of gravity
        return [ x0, x1, y0, y1, X, Y, A, FrameHeader[0] , FrameHeader[1] ]
    
    
    
    def ReadSingleBlob_float_coordinates(self , File, FrameHeader):
        '''
        will read the blob data from a blob file that was written
        in the "NEW style" with floating number coordinates
        '''
        # lower 8 bytes:
        x0 = (unpack('<h', File.read(2))[0]) # bounding box x0
        x1 = (unpack('<h', File.read(2))[0]) # bounding box x1
        y0 = (unpack('<h', File.read(2))[0]) # bounding box y0
        y1 = (unpack('<h', File.read(2))[0]) # bounding box y1
        
        # higher 8 bytes:
        higher_bites = unpack('<Q',File.read(8))[0]
        A = (2**20-1) & higher_bites         # Area
        X = (((0x00000fffff000000) & higher_bites) >> 24 ) / 256.0
        Y = (((0xfffff00000000000) & higher_bites) >> 44 ) / 256.0           # y center of gravity
        return [ x0, x1, y0, y1, X, Y, A, FrameHeader[0] , FrameHeader[1] ]
    
    
    
    def ReadBlobFile(self,fName, FrameStart = None, FrameEnd = None, FloatCoords = True):
        '''
        will go over the file (=fname) and store the frames 
        it containes in a the self.Blobs DataFrame
        
        fName - string, file to extract blob data from
        FrameStart - int, frame number at which to begin extracting blob data
        FrameEnd - int, frame number at which to stop extracting blob data
        '''
        if FloatCoords:
            readBlob = self.ReadSingleBlob_float_coordinates
        else:
            readBlob = self.ReadSingleBlob
        
        f = open(fName, "rb")
        self.fname = fName
        self.FrameCount += unpack('<i', f.read(4))[0]
        print 'total Frame number: %d'%self.FrameCount        
        
        # time in [day,mounth,year,hour,minute,sec,msec]:
        for j in [self.tStart, self.tEnd]:
            for i in range(7):
                j[i] = unpack('<i', f.read(4))[0]
        
        # dispose unwanted frames:

        if FrameEnd == None or FrameEnd > self.FrameCount:
            FrameEnd = self.FrameCount
        if FrameStart == None:
            FrameStart = 0          
        
        for i in range(FrameStart):
            #frame header
            head = []
            for j in range(self.FrameHeaderSize):
                head.append(unpack('<i', f.read(4))[0])
            #blobs:
            #print 'N Blobs: ' + str(head[2])  
            for j in range(head[2]):
                [f.read(2) for i in range(4)]
                f.read(4)
                [f.read(2) for i in range(2)]
        
        
        # get blobs of wanted frames:
        
        # for each frame in  the file:
        blobsSighted = []
        cycles = FrameEnd - FrameStart
        for i in range(cycles):
            #get frame header = [timeStamp, Frame number, blob count]:
            head = []
            for j in range(self.FrameHeaderSize):
                head.append(unpack('<i', f.read(4))[0])
            #get blobs:
            
            bin_size = 100000 
            counter = 0
            for j in range(head[2]):
                blobsSighted.append(readBlob(f, head))
                counter += 1                
                if counter >= bin_size:
                    temp = DataFrame(blobsSighted, columns=['x0','x1','y0','y1','Xcog','Ycog','Area','time','Frame'])
                    self.Blobs = self.Blobs.append(temp, ignore_index=True)
                    counter = 0
                    blobsSighted = []
                    
        temp = DataFrame(blobsSighted, columns=['x0','x1','y0','y1','Xcog','Ycog','Area','time','Frame'])
        self.Blobs = self.Blobs.append(temp, ignore_index=True)


    def SaveCsv(self, fname = None):
        '''
        use this in order to save the blobs data in this Reader
        as comma separated values file.
        '''
        if fname == None:
            extention = self.fname[:-4] + '_BlobsOut'
        else: 
            extention = fname
        self.Blobs.to_csv(path_or_buf = extention, sep=',')
                

    def SaveTargets(self, baseFname = None, decimals = 5, start_count=0):
        '''
        use this method to generate PyPTV comatible target files from 
        the blobs data of the BlobRecorder.
        
        start_count - in case it is needed to change the nobering of the
                      frames. for example to signal different runs.
                      if set to 0, then the frame numbers are unchanged.
                      the new frame number will be: 
                      original_number + start_count
                      
        '''
        if len(str(start_count)) > decimals:
            decimals = len(str(start_count))
            
        directory = os.path.join(os.getcwd() , 'Target_Files')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        if baseFname == None:
            baseFname = os.path.join(directory , self.fname[:5])
            
        formating = '%0.' + str(decimals) + 'd'
            
        for f in self.Blobs.Frame.unique():
            ff = f + start_count
            fname = baseFname + '.' + formating%ff + '_targets'         
            frameData = self.Blobs[self.Blobs.Frame == f]
            frameData.index = range(len(frameData))
            
            x = list(frameData.Xcog)
            y = list(frameData.Ycog)
            A = [9 for j in list(frameData.Area)]
            sumOfg = [9 for j in np.ones(len(x))]
            nx = [3 for j in list(abs(frameData.x0 - frameData.x1))]
            ny = [3 for j in list(abs(frameData.y0 - frameData.y1))]
            tnr = list(-1 * np.ones(len(x), dtype=np.int))
            '''
            A = [int(j) for j in list(frameData.Area)]
            sumOfg = [int(j) for j in np.ones(len(x))]
            nx = [int(j) for j in list(abs(frameData.x0 - frameData.x1))]
            ny = [int(j) for j in list(abs(frameData.y0 - frameData.y1))]
            tnr = list(-1 * np.ones(len(x), dtype=np.int))
            '''
            TargetFile = open(fname,'w')
            TargetFile.write(str(len(x)) + '\n')
            for j in range(len(x)):
                TargetFile.write('%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n'%(j,x[j],y[j],A[j],sumOfg[j],nx[j],ny[j],tnr[j]))
            TargetFile.close()
        
        frames = self.Blobs.Frame.unique()
        if len(frames) > 0:
            frameFullRange = range(int(frames[0]),int(frames[-1]) + 1)
        else:
            return
        black_target_string = '0    0.0    0.0     9     9     3     3    -1'
        for i in frameFullRange:
            if i not in frames:
                fname = baseFname + '.' + formating%(i+start_count) + '_targets'
                TargetFile = open(fname,'w')
                TargetFile.write('1\n')
                TargetFile.write(black_target_string)
                TargetFile.close()
             
             
    def Blob_count(self):
        '''
        returns (frames,count) -
        frames is an ordered list of all the unique frames in this data
        count is the ordered number of blobs in each frames
        this is a nice thing to plot for visualyzing the data
        '''
        count = self.Blobs['Frame'].value_counts()
        frames = count.index
        return frames, np.array(count)

        
    def full_blob_count(self):
        '''
        similar to blob count, BUT will return 0 for
        frames whith no blobs at all. This method is much 
        slower...
        '''
        f,c = self.Blob_count()
        mn, mx = min(f), max(f)
        full_f = np.arange(mn,mx+1)
        full_C = np.zeros_like( full_f )
        z = sorted(zip(f,c))
        for i in range(len(full_f)):
            if full_f[i] == z[0][0]:
                full_C[i] = z[0][1]
                z.remove(z[0])
                if len(z) == 0:
                    break
        return full_f,full_C
    
        
    def BlobFilterFrames(self,f_low = None, f_high=None):
        '''
        will remove all the blobs that is not in the frame range 
          f_low < frame < f_high
        '''
        if f_low != None:
            self.Blobs = self.Blobs[self.Blobs.Frame > f_low]
        if f_high != None:
            self.Blobs = self.Blobs[self.Blobs.Frame < f_high]    
        self.FrameCount = len(self.Blobs.Frame.unique())
        
        
    def Frame2Image(self, Frame, w=2304, h=1720):
        '''
        will create a PIL image of the frame F, with black bg and white blobs
        then you can use img.show() to plot or img.save() to save the image
        '''
        data = np.ones((h,w), dtype = 'int32')
        index = self.Blobs[self.Blobs.Frame == Frame].index
        for i in index:
            rx = range(int(self.Blobs.x0[i]), int(self.Blobs.x1[i] + 1))
            ry = range(int(self.Blobs.y0[i]), int(self.Blobs.y1[i] + 1))
            for x in rx:
                for y in ry:
                    data[y,x] = 255
        img = Image.fromarray(data)
        return img
    
    
    def plot_Frame_range(self, Frame_range, w=2304, h=1720, figsize=(8,8)):
        '''
        this function will create a folder figs. in this folder it will save 
        images corresponding to the blobs found, where blobs are seen as white
        squares over a black background.
        
        Frame_range - a list of integer frame numbers (e.g [100,101,102,..])
        w,h - the width and height of the frames in pixels (camera resolution)
        figsize - the size of the figure in inches
        '''
        plt.ioff()
        ld = os.listdir(os.getcwd())
        if 'figs' not in ld:
            os.mkdir('figs')   # create folder if there is none existing
        
        num_of_digits = 0
        n, tstnum = 1, Frame_range[-1]
        while tstnum/n != 0:
            n = n*10     # determine the number of digits
            num_of_digits+=1
        s = 'figs/%0.'+str(num_of_digits)+'d.tif' # filename format
        
        for f in Frame_range:
            img = self.Frame2Image(f,w=w,h=h)
            fig = plt.figure(figsize=(8,8))
            plt.imshow(img)
            fig.savefig(s%f)
            plt.close('all')
            
        plt.ion()
            
            




class extractor(object):
    '''
    class for unpacking and reading blob data to efficiently generate
    the target files from the wanted frames
    
    each extractor has a number of blobReader objects, where the extractor 
    uses them in an efficient manner.
    
    directions:
    1) load tha data with .load
    2) plot frames count with .plot_frame_count() to choose wanted frames
    3) save target files with .gen_Target_Files( f_start , f_last )
    
    - directory is the path where the blob files are located
    - blobFiles is a list of blobs file names
    - coord_format (= float or int) is the format of coordinates used in 
      recording (depends of the version of blobRecorder used)
    '''
    
    
    def __init__(self, directory, blobFiles, coord_format = float):
        self.dir = directory
        self.blbFls = blobFiles
        self.n = len(self.blbFls)
        self.readers = []
        for i in range(self.n):
            self.readers.append(blobReader())
        self.coord_fmt = coord_format
    
    
    def __repr__(self):
        s = 'blob extractor from: %s \n \n'%(self.dir)
        for i in range(self.n):
            f0,fn = int(min(self.readers[i].Blobs['Frame'])) , int(max(self.readers[i].Blobs['Frame']))
            s = s + '%s :  %d -- %d \n'%(self.blbFls[i], f0, fn)
        #print s
        return s

    
    
    def load(self ,  FrameStart = None, FrameEnd = None):
        os.chdir(self.dir)
        
        if self.coord_fmt == float:
            FloatCoords = True
        elif self.coord_fmt == int:
            FloatCoords = False
            
        for i in range(self.n):
            print 'unpacking blob%d'%i + '...'
            self.readers[i].ReadBlobFile(self.blbFls[i], FrameStart ,
                        FrameEnd, FloatCoords = FloatCoords)
      
        
    def plot_frame_count(self):
        '''
        plot the nmber of blobs seen at each frame
        '''
        color = ['b','r','g','y','k']
        fig,ax = plt.subplots()
        for i in range(self.n):
            f,c = self.readers[i].Blob_count()
            ax.scatter(f,c,c=color[i],label='reader%d'%i)
        ax.legend()
        ax.set_xlabel('frame #')
        ax.set_ylabel('number of blobs')
            
            
    def plot_frames(self,f_start,f_last):
        '''
        make a scatter plot of the blobs that were 
        seen in frame    f_start<f<f_last 
        '''
        color = ['b','r','g','y','k']
        fig,ax = plt.subplots()
        for i in range(self.n):
            a = self.readers[i].Blobs[self.readers[i].Blobs.Frame > f_start]
            a = a[a.Frame < f_last]
            ax.scatter(a.Xcog, a.Ycog,c=color[i],label='reader%d'%i)
        ax.legend()
        ax.set_xlabel('frame #')
        ax.set_ylabel('number of blobs')
                    
            
    def gen_Target_Files(self,f_start,f_last, decimals=4, start_count=0):
        '''
        will generate the pyPTV compatible target files for the
        frames such that    f_start <= f <= f_last
        '''
        
        if len(str(start_count)) > decimals:
            decimals = len(str(start_count))
        
        formating = '%0.' + str(decimals) + 'd'
        
        for i in range(self.n):
            baseFname = 'blob%d'%i
            a = self.readers[i].Blobs[self.readers[i].Blobs.Frame >= f_start]
            a = a[a.Frame <= f_last]
            temp = blobReader()
            temp.Blobs = a
            temp.fname = baseFname
            temp.SaveTargets( decimals = decimals, start_count = start_count)
            
            saved_frames = temp.Blobs.Frame.unique()
            if len(saved_frames)>0:
                add_frames_list = [range(f_start,int(saved_frames[0])),
                                   range(1 + int(saved_frames[-1]),f_last+1)]
            else:
                add_frames_list = [range(f_start, f_last+1), []]
            directory = os.path.join(os.getcwd() , 'Target_Files' , baseFname)
            black_target_string = '0    0.0    0.0     9     9     3     3    -1'
            for lst in add_frames_list:
                for f in lst:
                    fname = directory + '.' + formating%(f+start_count) + '_targets'
                    TargetFile = open(fname,'w')
                    TargetFile.write('1\n')
                    TargetFile.write(black_target_string)
                    TargetFile.close()
                        
        l = os.listdir(os.path.join(os.getcwd() , 'Target_Files'))
        b = []
        for i in range(len(self.readers)):
            b.append([])
        k = 6 + decimals
        for targ in l:
            b[int(targ[4])].append(int(targ[6:k]))
        print 'finished saving targets! start and end frames are:'
        m1,m2 = [],[]
        for i in b:
            m1.append(min(i))
            m2.append(max(i))
        print max( m1 ) , min( m2 )
        
        
    def get_potential_good_frames(self):
        '''
        will return a list of tuples (f_start,f_end), where
        f_start and f_end signal to frame sequences with blobs
        that were sighted by at least two blob readers.
        '''
        mn, mx = 1e500,0
        for i in self.readers:
            t = min(i.Blobs['Frame']), max(i.Blobs['Frame'])
            if t[0]<mn: mn = t[0]
            if t[1]>mx: mx = t[1]
        frm = range(int(mn), int(mx)+1)
        time = 0.001*(mx-mn)
        print 'time estimate ~ %0.1f seconds'%time
        potential = []
        for i in frm:
            k, n = 0, 0
            for j in self.readers:
                l = len(j.Blobs[j.Blobs['Frame']==i])
                if l>0:
                    k+=1
                    n+= l
            potential.append( (k>=2) * n)
        return (frm,potential)
                
    
    def getFrameRange(self):
        '''
        plot the nmber of blobs seen at each frame
        '''
        mi,mx = sys.maxint , 0
        for i in range(self.n):
            f,c = self.readers[i].Blob_count()
            f = list(f)
            if np.amax(f) > mx: mx = int(np.amax(f))
            if np.amin(f) < mi: mi = int(np.amin(f))
        return mi , mx
        
        

    
def bits_from_file(N, File):
    '''will read N bytes from File and return 
    the data in bits'''
    h_bytes = []
    for i in range(N):
        h_bytes.append(File.read(1))
    h_bits = ''
    for b in h_bytes:
        bits = bin(ord(b))[2:].rjust(8, '0')
        for d in range(len(bits)-1,-1,-1):
            h_bits += bits[d]
    return h_bits
    





def bits2int(bits):
    '''return an integer from a string thatrepresents
       a binary number '''
    bits = [int(x) for x in bits[::-1]]
    x = 0
    for i in range(len(bits)):
        x += bits[i]*2**i
    return x




    
def ReadSingleBlob_float_coordinates(self , File, FrameHeader):
    '''
    will read the blob data from a blob file that was written
    in the "NEW style" with floating number coordinates
    '''
    # lower 8 bytes:
    x0 = (unpack('<h', File.read(2))[0]) # bounding box x0
    x1 = (unpack('<h', File.read(2))[0]) # bounding box x1
    y0 = (unpack('<h', File.read(2))[0]) # bounding box y0
    y1 = (unpack('<h', File.read(2))[0]) # bounding box y1
    
    # higher 8 bytes:
    higher_bit = bits_from_file(8, File)
    A = bits2int(higher_bit[:20])         # Area
    x_frac = bits2int(higher_bit[24:32])
    x_int = bits2int(higher_bit[32:44])
    y_frac = bits2int(higher_bit[44:52])
    y_int = bits2int(higher_bit[52:64])
    X = x_int + x_frac/256.0            # x center of gravity
    Y = y_int + y_frac/256.0            # y center of gravity
    return [ x0, x1, y0, y1, X, Y, A, FrameHeader[0] , FrameHeader[1] ]
   


     
def ReadSingleBlob(File):
        x0 = (unpack('<h', File.read(2))[0]) # bounding box x0
        x1 = (unpack('<h', File.read(2))[0]) # bounding box x1
        y0 = (unpack('<h', File.read(2))[0]) # bounding box y0
        y1 = (unpack('<h', File.read(2))[0]) # bounding box y1
        A = (unpack('<i', File.read(4))[0])  # Area
        X = (unpack('<h', File.read(2))[0])  # x center of gravity
        Y = (unpack('<h', File.read(2))[0])  # y center of gravity
        return [ x0, x1, y0, y1, X, Y, A ]

