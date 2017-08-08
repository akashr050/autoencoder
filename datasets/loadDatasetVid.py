""" Function to load datasets on the go (for expt 4 and beyond only!!!)"""

import cv2
import os
import numpy as np
import h5py
from vggPreprocessing_np import preprocess_image

def _window_function(Data, Label, Type, k):
    """ 
    :type  Data: numpy array (4D)
    :param Data: Multi-frame input

    :type  Label: numpy array (1D) 
    :param Label: Label of length same as number of frames in Data

    :type  Type: string
    :param Type: Type of video input (All frames, Sliding window: Kframes and 
                 Sliding window with overlap: Kframes at K/2 overlap)

    :type  k: int 
    :param k: Integer number denoting sliding window size

    """ 
    # K frame sliding window without any overlap
    if Type == 'kframes' or Type == 'kframespad':

        Data  = Data[:(Data.shape[0]/k) * k, :, :, :]
        Label = Label[:(Label.shape[0]/k) * k]

        
    # END IF

    return Data, Label

def _fft_function(Data, Label, Type):
    """ 
    :type  Data: numpy array (3D or 4D)
    :param Data: Multi-frame input

    :type  Label: numpy array (1D) 
    :param Label: Label of length same as number of frames in Data

    :type  Type: string
    :param Type: Type of video input (All frames, Sliding window: Kframes and 
                 Sliding window with overlap: Kframes at K/2 overlap)

    """ 
    if Type == 'kframes':
        # FFT conversion of data
        frames_, channels_, height_, width_ = Data.shape
        tempData  = []
        finalData = []

        for item in range(Data.shape[0]/k):
            tempData  = np.transpose(Data[(item*k):(item*k)+k,:,:,:], (1,2,3,0))
            tempData  = tempData.reshape(height_*width_*channels_, k)
            tempData  = np.hstack((np.fft.fft(tempData,n=32).real, np.fft.fft(tempData,n=32).imag))
    	    tempData  = tempData.reshape(height_, width_, channels_, 64) # 64-pt FFT
            tempData  = np.transpose(tempData, (3,0,1,2))
            if len(finalData) == 0:
                finalData = tempData
            else:
                finalData = np.vstack((finalData, tempData))

            # END IF

        # END FOR

        Data = finalData
        del finalData
        del tempData

    # END IF

    return Data, np.repeat(Label, Data.shape[0])


def _load_dataset(vidsFile, baseDataPath, index, frameTotal, size, isTraining, classIndFile='/z/home/madantrg/Datasets/FilesUCF101Orig/classInd.txt', chunk=100, Type='normal', k=20, FFT=0):

    """ 
    :type  vidsFile: string
    :param vidsFile: '.txt' file providing a list of videos with the full path to the video 
                     included and its label

    :type  baseDataPath: string
    :param baseDataPath: Full path to the folder contain the original videos (dataset)

    :type  index: int  
    :param index: Index value of video number
   
    :type  fName: string
    :param fName: Prefix of HDF5 (train/val/test list)

    :type  frameTotal: int
    :param frameTotal: Minimum number of frames

    :type  size: int
    :param size: Output size of each frame

    :type  isTraining: bool
    :param isTraining: Boolean to indicate load operation is for training or evaluation 

    :type  classIndFile: string
    :param classIndFile: Full path to file containing class list and labels for a given dataset

    :type  chunk: int
    :param chunk: Number of videos within a single HDF5 file 

    :type  Type: string
    :param Type: Type of video input (All frames, Sliding window: Kframes and 
                 Sliding window with overlap: Kframes at K/2 overlap)
    :type  k: int
    :param k: Number of frames within a sliding window 
    
    :type FFT: int
    : param FFT: Binary value to generate FFT input or provide raw pixel values

    """
    
    # Read contents of video file
    fin         = open(vidsFile,'r+')
    lines       = fin.readlines()

    # Initialize values in case loading data fails
    FAIL_FLAG   = False
    Data        = np.zeros((1,1))
    Label       = np.zeros((1,1))

    try: 
        Data        = []
        Label       = []

        # 1. Load Data from video 
        Data        = []
        vid         = cv2.VideoCapture(os.path.join(baseDataPath, os.path.join(os.path.split(lines[index])[0], os.path.split(lines[index])[1].split(' ')[0])).replace('\r','').replace('\n',''))
        count       = 0

        flag, frame = vid.read()

        while flag: 
            
            H,W,C = frame.shape

            if count == 0:
                Data = frame.reshape(1, H, W, C)
            else:
                Data = np.concatenate((Data, frame.reshape(1, H, W, C)))

            count += 1
            flag, frame= vid.read()

        # 2. Generate label 
        try:
            Label  = np.zeros((Data.shape[0],)) + int(os.path.split(lines[index])[1].split(' ')[1]) 

        except:
            # Fail safe to obtain indices if test data with no labels is provided
            CLASS = os.path.split(lines[index].split(' ')[0])[0]
            if len(Label) == 0:
                fIn = open(classIndFile,'r')
                classLines = fIn.readlines()
                Label = np.repeat([int(x.split(' ')[0])-1 for x in classLines if CLASS in x.split()], Data.shape[0])

        # END TRY
        Label = Label.astype(dtype= 'int32')
       
        # 3. If the total number of frames < minimum number of frames required, interpolate data
        if Data.shape[0] < frameTotal or Data.shape[0] < 2*k:
            if frameTotal > 2*k:
                indices = np.linspace(0, Data.shape[0], frameTotal)  
                tempData = np.zeros((frameTotal, Data.shape[1], Data.shape[2], 3), np.float32)  
            else:
                indices = np.linspace(0, Data.shape[0], 2*k)  
                tempData = np.zeros((2*k, Data.shape[1], Data.shape[2], 3), np.float32)  

            for row in range(Data.shape[1]):
                for col in range(Data.shape[2]):
                    tempData[:,row,col,0] = np.interp(indices, range(Data.shape[0]), Data[:,row,col,0].astype('float32'))
                    tempData[:,row,col,1] = np.interp(indices, range(Data.shape[0]), Data[:,row,col,1].astype('float32'))
                    tempData[:,row,col,2] = np.interp(indices, range(Data.shape[0]), Data[:,row,col,2].astype('float32'))

            Data = tempData
            Label = np.repeat(Label[0], Data.shape[0])
            del tempData

        #END IF

        # 4. Amplitude normalization 
        #Data  = Data/255.

        # 5. Apply sampling or sliding window appropriately
        if Type == 'normal':
            Data = Data[np.linspace(0, Data.shape[0]-1,frameTotal).astype('int32'),:,:,:].astype('float32')
            Label = np.repeat(Label[0],Data.shape[0])

        elif Type == 'full':
            pass

            

        else:
            Data, Label = _window_function(Data, Label, Type, k)

        # END IF

        # 6. Apply FFT if necessary
        if FFT:
            Data, Label = _fft_function(Data, Label, FFT, Type)

        # 7. If frame exists, pre-processing
        tempData = np.zeros((Data.shape[0],size,size,3))
        for idx in range(Data.shape[0]):
            tempData[idx,:,:,:] = preprocess_image(Data[idx,:,:,:], size, size, isTraining)
            

        # END FOR

        Data = tempData
        del tempData

        # Ensure that all videos are padded with 0s to match 300 frame length
        if Type == 'kframespad':
            if Data.shape[0] < 300:
                Data  = np.vstack((Data,np.zeros((300 - Data.shape[0],size,size,3))))
                Label = np.repeat(Label[0],300)

    except:
        print "Failed index is: ", index
        print "Total number of videos in list is: ", len(lines)
        print "Failed video is: ", lines[index]

        # Force failure of cluster job is loading video fails
        import pdb; pdb.set_trace()

        FAIL_FLAG = True

    # END TRY


    # Deallocate
    del fin
    del lines

    return Data, Label, FAIL_FLAG


if __name__=="__main__":

    D, L, FLG = _load_dataset('/z/home/madantrg/Datasets/FilesHMDB51Rate/trainlist01.txt','/z/home/madantrg/Datasets/HMDB51Rate/', 6781, 16, 224, False, '/z/home/madantrg/Datasets/FilesHMDB51Rate/classInd.txt')
    import pdb; pdb.set_trace()
