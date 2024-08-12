import os
import glob
import numpy as np
import nibabel as nib
import splitfolders
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

list1 = sorted(glob.glob('TrainingData/raw/*/*t2.nii'))
list2 = sorted(glob.glob('TrainingData/raw/*/*t1ce.nii'))
list3 = sorted(glob.glob('TrainingData/raw/*/*flair.nii'))
list4 = sorted(glob.glob('TrainingData/raw/*/*seg.nii'))

os.mkdir('TrainingData/processed')
os.mkdir('TrainingData/raw/masks')
os.mkdir('TrainingData/raw/images')

for img in range(len(list1)):
    print("Now preparing image and masks number: ", img)

    img1=nib.load(list1[img]).get_fdata()
    img1=scaler.fit_transform(img1.reshape(-1,img1.shape[-1])).reshape(img1.shape)

    img2=nib.load(list2[img]).get_fdata()
    img2=scaler.fit_transform(img2.reshape(-1,img2.shape[-1])).reshape(img2.shape)

    img3=nib.load(list2[img]).get_fdata()
    img3=scaler.fit_transform(img3.reshape(-1,img3.shape[-1])).reshape(img3.shape)

    imgs=np.stack([img3,img2,img1],axis=3)
    imgs=imgs[56:184, 56:184, 13:141]

    mask=nib.load(list4[img]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask==4]=3
    mask=mask[56:184, 56:184, 13:141]
    mask=to_categorical(mask, 4)

    np.save('TrainingData/raw/images/image_'+str(img)+'.npy',imgs)
    np.save('TrainingData/raw/masks/mask_'+str(img)+'.npy',mask)

inp='TrainingData/raw/'
out='TrainingData/processed/'

splitfolders.ratio(inp, output=out, seed=42, ratio=(.75, .25), group_prefix=None)
os.rename('TrainingData/processed/val','TrainingData/processed/valid')