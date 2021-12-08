#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Neural network layers from 
# #### https://keras.io/examples/vision/conv_lstm/

#### Functions to load data

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict
#import torchvision.transforms as T
from datetime import datetime
from tensorflow import keras


# In[2]:




"""
Function for loading the satellite images
Arguments:
 selectedDataset - can be "2019-07" - e.g. specific month
                          "" - all months
 pictureTypes - list defining what sort of constellations are loaded
                "dnc" - 
                "dnm" - 24-hour Microphysics RGB

 pictureSize - Size to resize images to after they are read from disk. Defaults to (256, 256).
Returns:
  2 dictionaries
  dates - dates corresponding to pictures
  pictures - satellite pictures
"""


def loadSatelliteImages(selectedDatasets=["2019-07"], pictureTypes=["dnc", "dnm"], pictureSize=(256, 256)):
    pictures = defaultdict(lambda: defaultdict(list))
    dates = defaultdict(lambda: defaultdict(list))

    for selectedDataset in selectedDatasets:
        for pictureType in pictureTypes:
            satellitePictureNames = os.listdir(os.path.join(".", "data", selectedDataset, pictureType))
            pictures[selectedDataset][pictureType] = []
            dates[selectedDataset][pictureType] = []
            for satellitePictureName in satellitePictureNames:
                # Load image
                imageDateStr = satellitePictureName.replace("dnc-", "").replace("dnm-", "").replace(".png", "")
                #Parse date to datetime 2019-07-01-05-45
                imageDate = datetime.strptime(imageDateStr, "%Y-%m-%d-%H-%M")
                img = image.load_img(os.path.join(".", "data", selectedDataset,
                                                  pictureType, satellitePictureName),
                                     target_size=pictureSize)
                # Convert to np array and add to list
                pictures[selectedDataset][pictureType].append(np.array(img))
                dates[selectedDataset][pictureType].append(imageDate)
            dates[selectedDataset][pictureType] = dates[selectedDataset][pictureType])
            pictures[selectedDataset][pictureType] = np.array(pictures[selectedDataset][pictureType])
            #Argsort
            sortedDates = np.argsort(dates[selectedDataset][pictureType])
            dates[selectedDataset][pictureType] = dates[selectedDataset][pictureType][sortedDates]
            pictures[selectedDataset][pictureType] = pictures[selectedDataset][pictureType][sortedDates]

    return pictures, dates


# In[3]:


npixel = 128


# In[4]:



sat, labels = loadSatelliteImages(selectedDatasets=["2019-06","2019-07", "2019-08", "2019-09", "2019-09"], pictureTypes=["dnc"], pictureSize=(npixel, npixel))


# In[5]:


#plt.imshow(sat["2019-07"]["dnc"][30])


# ## Transforming images to dataset

# In[6]:


"""
Function for creating samples out of satellite data. Each row of X contains given number 
(imagesInSample) used to predict future weather. Rows of y are similar to y, although shifted
by one time interval

Arguments
dataDict - dictionary, which contains image data
X_imagetype - the type of images that are requested for X
Y_imagetype - the type of images that are requested for y
imagesInSample - number of images in data row
"""
def createDataSetFromImages(dataDict, X_imagetype, Y_imagetype, selectChannelX=None, selectChannelY=None, imagesInSample=6):
    X = []
    y = []
    for month in dataDict.keys():
        if selectChannelX is None:
            X_subset = dataDict[month][X_imagetype]
        else:
            X_subset = dataDict[month][X_imagetype][:,:,:,selectChannelX]
            # Add a channel dimension if using only one channel
            X_subset = np.expand_dims(X_subset, axis=-1)

        if selectChannelY is None:
            y_subset = dataDict[month][Y_imagetype]
        else:
            y_subset = dataDict[month][Y_imagetype][:,:,:,selectChannelY]
            y_subset = np.expand_dims(y_subset, axis=-1)
        assert len(X_subset)==len(y_subset) # Lengths must match
        for i in range(0, len(X_subset)-imagesInSample-1):
            #Select images so that y is shifted by one frame
            selected_X = X_subset[i:i+imagesInSample]
            selected_y = y_subset[i+1:i+1+imagesInSample]
            X.append(selected_X)
            y.append(selected_y)
    return np.array(X), np.array(y)


# In[7]:


# Normalize the data to the 0-1 range.
for dataSet in sat.keys():
    for pictureType in sat[dataSet].keys():
        sat[dataSet][pictureType] = sat[dataSet][pictureType] / 255


# In[8]:

#From empirical testing found that for later correlation, DNC channel 1 is the best
X, y = createDataSetFromImages(sat, "dnc", "dnc", 1, 1)


# In[9]:


print(f"Check that shift is OK, following value must be 0:{np.sum(y[-1][0]-X[-1][1])}")
print(f"Check that shift is OK, following value must be 0:{np.sum(y[0][0]-X[0][1])}")


# In[10]:




# Split into train and validation sets
indexes = np.arange(X.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.8 * X.shape[0])]
val_index = indexes[int(0.8 * X.shape[0]) :]
train_X = X[train_index]
train_y = y[train_index]
val_X = X[val_index]
val_y = y[val_index]


# In[11]:


print(f"Check that shift is OK, following value must be 0:{np.sum(val_y[-1][0]-val_X[-1][1])}")
print(f"Check that shift is OK, following value must be 0:{np.sum(val_y[0][0]-val_X[0][1])}")


# In[12]:


#Check that dims match
print("Training Dataset Shapes: " + str(train_X.shape) + ", " + str(train_y.shape))
print("Validation Dataset Shapes: " + str(val_X.shape) + ", " + str(val_y.shape))


# # Start building model
# 
# #### https://keras.io/examples/vision/conv_lstm/
# 
# #### https://github.com/xibinyue/ConvLSTM-1/blob/master/radar_forecast.py

# In[13]:


from tensorflow.keras import layers
from keras.models import Sequential


# In[15]:


# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *train_X.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)


# In[16]:


# Build model
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
)


# In[17]:


model.summary()


# ## Train model

# In[18]:


# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 12
batch_size = 10

# Fit the model to the training data.
model.fit(
    train_X,
    train_y,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_X, val_y),
    callbacks=[early_stopping, reduce_lr],
)


# In[21]:


model.save('savedModel_2a')

