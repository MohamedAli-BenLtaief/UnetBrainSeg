import sys,os,io
import keras
import numpy as np
from unet import model
from matplotlib import pyplot as plt
import segmentation_models_3D as sm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load(imgdir, imglist):
    images=[]
    for i,img in enumerate(imglist):    
        if (img.split('.')[1] == 'npy'):
            img = np.load(imgdir+img).astype(np.float32)
            images.append(img)
    images = np.array(images)
    return(images)

def imageLoader(imgdir, imglist, mskdir, msklist, size):
    n = len(imglist)
    while True:
        s, e = 0, size
        while s < n:
            x = load(imgdir, imglist[s:min(e, n)])
            y = load(mskdir, msklist[s:min(e, n)])
            yield (x, y)
            s += size
            e += size

timgdir = "TrainingData/processed/train/images/"
tmskdir = "TrainingData/processed/train/masks/"
vimgdir = "TrainingData/processed/valid/images/"
vmskdir = "TrainingData/processed/valid/masks/"
timglist = os.listdir(timgdir)
tmsklist = os.listdir(tmskdir)
vimglist = os.listdir(vimgdir)
vmsklist = os.listdir(vmskdir)
timgdatagen = imageLoader(timgdir, timglist, tmskdir, tmsklist, 2)
vimgdatagen = imageLoader(vimgdir, vimglist, vmskdir, vmsklist, 2)

#Training
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + focal_loss
metrics = ['accuracy', sm.metrics.FScore(threshold=0.5), sm.metrics.IOUScore(threshold=0.5)]
optim = keras.optimizers.Adam(0.0001)

batchsize=2
tsteps=len(timglist)//batchsize
vsteps=len(vimglist)//batchsize
model = model(128, 128, 128, 3, 4)
model.compile(optimizer = optim, loss = total_loss, metrics = metrics)
history=model.fit(timgdatagen,steps_per_epoch=tsteps,epochs=100,verbose=1,validation_data=vimgdatagen,validation_steps=vsteps)
os.mkdir('SavedModels')
model.save('SavedModels/brats.hdf5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Testing
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score

my_model = load_model('SavedModels/brats.hdf5', compile=False)

test_img_datagen = imageLoader(vimgdir, vimglist, vmskdir, vmsklist, 8)
test_image_batch, test_mask_batch = test_img_datagen.__next__()
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4).flatten()
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4).flatten()

mean_f1_score = f1_score(test_mask_batch_argmax, test_pred_batch_argmax, average='macro')
print("Mean F1 Score =", mean_f1_score)