#!/usr/bin/env python
# coding: utf-8

# # Notebook to correlate different channels in satellite images and measured sun intensity from weather stations

# In[1]:


#### Functions to load data

from keras.preprocessing import image
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict
from datetime import datetime
from tensorflow import keras
from PIL import Image
#from matplotlib.patches import Circle
from datetime import date
from tensorflow import keras
import tensorflow as tf
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ## Locations of weather stations

# In[2]:


weather_station_coordinates = dict()
weather_station_coordinates["Tallinn-Harku"] = [59.398055, 24.602778]
weather_station_coordinates["Narva"] = [59.389444, 28.109167]
weather_station_coordinates["Pärnu"] = [58.384556, 24.485197]
weather_station_coordinates["Roomassaare"] = [58.218056, 22.506389]
weather_station_coordinates["Tartu-Tõravere"] = [58.264167, 26.461389]
weather_station_coordinates["Tiirikoja"] = [58.865278, 26.952222]
weather_station_coordinates["Vilsandi"] = [58.382778, 21.814167]


# ## Load satellite images

# In[3]:




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
            dates[selectedDataset][pictureType] = np.array(dates[selectedDataset][pictureType])
            pictures[selectedDataset][pictureType] = np.array(pictures[selectedDataset][pictureType])
            #Argsort
            sortedDates = np.argsort(dates[selectedDataset][pictureType])
            dates[selectedDataset][pictureType] = dates[selectedDataset][pictureType][sortedDates]
            pictures[selectedDataset][pictureType] = pictures[selectedDataset][pictureType][sortedDates]

    return pictures, dates




###### Some important variables
 
npixel = 128 # Define the pixel size of images
pictureType = ["dnc", "dnm"]
dataSets = ["2019-06","2019-07","2019-08","2019-09","2019-10"]

#####################################
# Start loading satellite images ####
#####################################


# Function to relate points in satellite image to weather station coordinates
def findPixel(latsMatrix, lonsMatrix, coordLat, coordLon):
    #Iterate over lattitude and longitude matrix to find coord
    accuracyDif = 0.07
    for i in range(len(latsMatrix)):
        for j in range(len(lonsMatrix[i])):
            currentLat = latsMatrix[i][j]
            currentLon = lonsMatrix[i][j]
            if(np.abs(currentLat-coordLat)<accuracyDif):
                if(np.abs(currentLon-coordLon)<accuracyDif):
                    return (j, i)
    #Nothing was found return
    return (None, None)


# Function to resize the transformation matrix, if the size of picture has been changed during reloading
def resizeLatsLonsMatrix(latsMatrix, lonsMatrix, targetSize):
    result = [latsMatrix, lonsMatrix]
    for i in range(len(result)):
        pil = Image.fromarray(result[i])
        resized = pil.resize((targetSize,targetSize))
        resizedArray = np.array(resized)
        result[i] = resizedArray
    return result


#Load sattelite images
sat, labels = loadSatelliteImages(selectedDatasets=dataSets, pictureTypes=pictureType, pictureSize=(npixel, npixel))

#The initial coordinate matrix is for images 650x650
lats650 = np.load("./OstlandA/OstlandA-lats.npy")
lons650 = np.load("./OstlandA/OstlandA-lons.npy")
#Convert them to size (npixel x npixel)
lats_npixel, lons_npixel = resizeLatsLonsMatrix(lats650, lons650, npixel)


# In[9]:


#Find the pixels where the weather stations are located
weather_station_loc_pixels = dict()
for station in weather_station_coordinates.keys():
    weather_station_loc_pixels[station]=findPixel(lats_npixel, lons_npixel, 
                          weather_station_coordinates[station][0], 
                          weather_station_coordinates[station][1])


# Normalize the of satellite images to the 0-1 range.
for dataSet in sat.keys():
    for selectedPictureType in sat[dataSet].keys():
        sat[dataSet][selectedPictureType] = sat[dataSet][selectedPictureType]/255

####################################################
# Load solar intensity data from weather stations ##
####################################################


# Some weather stations have changed locations over time, as the differences between their locations are rather small (less than 8 km)
# We will not make a separation between their data

# Function for joining columns, where an area has two weather measuring points
def join_columns(c1, c2, nc, df, column_id): 
    data = []
    cs = [c1, c2]
    for i, rows in df[cs].iterrows():
        if (pd.isna(rows[0]) == True) & (pd.isna(rows[1]) == False):
            data.append(round(rows[1], 2))
        elif (pd.isna(rows[0]) == False) & (pd.isna(rows[1]) == True):
            data.append(round(rows[0], 2))
        elif (pd.isna(rows[0]) == False) & (pd.isna(rows[1]) == False):
            data.append(round(rows.mean(), 2))
        elif (pd.isna(rows[0]) == True) & (pd.isna(rows[1]) == True):
            data.append(rows[0])

    df = df.drop(columns = [c1, c2])
    df.insert(column_id, nc, data)
    
    return df


# Function to datetime object column to dataframe from year, month and day
def createDateTimeColumn(df):
    dateTimes = []
    for i in range(len(df)):
        row = df.iloc[i]
        dateTimes+=[datetime.combine(date(row.y, row.m, row.d), row.time)]
    df["dateTime"] = dateTimes

# Function to shift the times -X hours to facilitate predicting future solar intensity from existing
from datetime import timedelta
import copy
def shiftDateTime(df, numberOfHours):
    dateTimes = []
    for i in range(len(df)):
        row = df.iloc[i]
        dateTimes+=[datetime.combine(date(row.y, row.m, row.d), row.time)+timedelta(hours=numberOfHours)]
    df2 = copy.deepcopy(df)
    df2["y"] = [date.year for date in dateTimes]
    df2["m"] = [date.month for date in dateTimes]
    df2["d"] = [date.day for date in dateTimes]
    df2["time"] = [date.time() for date in dateTimes]
    
    return df2

#Load initial data
hourly_sun_intensity = pd.read_excel('./data/2-10_21_524-2 Andmed.xlsx', sheet_name = 'tunni sum.kiirgus', header = 1)

#Update column names by shortening them and converting to English
newColumnNames = dict()
newColumnNames["Aasta"] = "y"
newColumnNames["Kuu"] = "m"
newColumnNames["Päaev"] = "d"
newColumnNames["Kell (UTC)"] = "time"

for columnName in hourly_sun_intensity.columns:
    if "kiirgus" in columnName:
        newColumnNames[columnName] = "solar_"+columnName.replace(" summaarne kiirgus, W/m²", "")
hourly_sun_intensity = hourly_sun_intensity.rename(columns=newColumnNames)


#Merge columns, which are due to weather station moving
hourly_sun_intensity = join_columns('solar_Narva', 'solar_Narva-Jõesuu', 'solar_Narva', hourly_sun_intensity, 4)
hourly_sun_intensity = join_columns('solar_Pärnu-Sauga', 'solar_Pärnu', 'solar_Pärnu', hourly_sun_intensity, 5)

#Drop rows where some value is missing
hourly_sun_intensity = hourly_sun_intensity.dropna()
#If value is -1 it corresponds to night, set it to 0
hourly_sun_intensity = hourly_sun_intensity.replace(-1, 0)

#Create datetime object column for finding matching satellite images and rows of weather station data
createDateTimeColumn(hourly_sun_intensity)

##Shift solar intensity time -1 h to allow predicting future
hourly_sun_intensity_shifted = hourly_sun_intensity#shiftDateTime(hourly_sun_intensity, -1)


# ### As satelite images are from 2019, we can drop other years
hourly_sun_intensity_shifted = hourly_sun_intensity_shifted[hourly_sun_intensity_shifted.y == 2019]


# ### Filter the data for matching
#Masks for selecting right rows of weather data and drop others

mask = np.asarray([hourly_sun_intensity_shifted["dateTime"].iloc[i] in labels[dataSets[0]][pictureType[0]] for i in range(len(hourly_sun_intensity_shifted["dateTime"]))])

for j in range(1,len(dataSets)):
    mask_aux = np.asarray([hourly_sun_intensity_shifted["dateTime"].iloc[i] in labels[dataSets[j]][pictureType[0]] for i in range(len(hourly_sun_intensity_shifted["dateTime"]))])
    mask = mask + mask_aux
hourly_sun_intensity_filtered = hourly_sun_intensity_shifted[mask]

#################################################
###### Create combined dataset for model ########
#################################################
#
#Function for creating samples out of satellite data. Each row of X contains given number 
#(imagesInSample) used to predict future weather. Rows of y are similar to y, although shifted
#by one time interval
def createDataSetFromImages(dataDict, imagetype, imageLabels, weatherDataLabels, selectChannelX=None, imagesInSample=6, skipImages=1):
    X = []
    X_times = []
    for month in dataDict.keys():
        if selectChannelX is None:
            X_subset = dataDict[month][imagetype]
        else:
            X_subset = dataDict[month][imagetype][:,:,:,selectChannelX]
            # Add a channel dimension if using only one channel
            X_subset = np.expand_dims(X_subset, axis=-1)


        X_times_subset = imageLabels[month][imagetype]
        assert len(X_subset)==len(X_times_subset) # Lengths must match
        for i in range(0, len(X_subset)-imagesInSample*skipImages-1):
            #Select only images, which last frame corresponds to time for which we
            #have solar intensity
            selected_X_future_time = X_times_subset[i+imagesInSample*skipImages]
            if(selected_X_future_time in weatherDataLabels):
                #Select images so that y is shifted by one frame
                selected_X = X_subset[i:i+imagesInSample*skipImages:skipImages]
                selected_X_times = X_times_subset[i:i+imagesInSample*skipImages:skipImages]
                X.append(list(selected_X))
                X_times.append(list(selected_X_times))
    return np.array(X), np.array(X_times)


#Select only satellite images as prediction targets, where the last frame of RNN is in Weather data table
labelsAsTImestamps = [pd.Timestamp(value) for value in hourly_sun_intensity_filtered.dateTime.values]
X_dnc, X_times_dnc = createDataSetFromImages(sat, "dnc", labels, labelsAsTImestamps, imagesInSample=8, skipImages=4)
X_dnm, X_times_dnm = createDataSetFromImages(sat, "dnm", labels, labelsAsTImestamps, imagesInSample=8, skipImages=4)


# ### Link together points on satellite images and weather data
X_dnc_filtered = X_dnc
X_dnm_filtered = X_dnm
X_times_filtered = X_times_dnc
X_filtered_images = dict()
X_filtered_images["dnc"] = X_dnc_filtered
X_filtered_images["dnm"] = X_dnm_filtered


# ### Transforming points on images and sunlight data to dataset

#Function for creating samples out of satellite data. Each row of X contains given number 
#(imagesInSample) used to predict future weather. Rows of y are sun intensities at selected location,
#although shifted
#by one time interval
def createDataSetFromImagesToFeatures(satelliteData, satelliteTimes, intensityData, 
                                       weatherStationPixels, howMuchToPredictAhead=1):
    X = []
    y = []
    intensityData = shiftDateTime(intensityData, -howMuchToPredictAhead)
    satelliteDataKeys = list(satelliteData.keys())
    numberOfSamples = len(satelliteData[satelliteDataKeys[0]])
    numberOfFrames = len(satelliteData[satelliteDataKeys[0]][0])
    #For each sample
    for i in range(numberOfSamples):
        #For each weather station
        for weatherStation in weatherStationPixels.keys():
            selectedStation = weatherStationPixels[weatherStation]
            X_subset = []
            y_subset = []
            #For each frame in sample
            for j in range(numberOfFrames):
                #Select the right row in solar intensity
                intensityDataRow = intensityData[intensityData.dateTime == satelliteTimes[i][j]] 
                X_sub_subset = []
                #For all picture types
                for picType in satelliteData.keys():
                    #Pick values from specific coordinates from image
                    X_sub_subset += list(satelliteData[picType][i][j][selectedStation[0], selectedStation[1]].flatten())
                X_subset+=[X_sub_subset]
                y_subset+=list(intensityDataRow[("solar_"+weatherStation)].to_numpy())
            if(len(y_subset)==len(X_subset)):
                y+=[np.array(y_subset)]
                X+=[X_subset]
       
    return np.array(X), np.array(y)


X_images_weather, y_intensity = createDataSetFromImagesToFeatures(X_filtered_images, X_times_filtered,
                                   hourly_sun_intensity_filtered, weather_station_loc_pixels)


# In[116]:

#Check if we have right number of y values in sample
for i in range(len(y_intensity)):
    if len(y_intensity[i])!=8:
        print("ERROR!")
        print(i)

##################################################
############# Train models #######################
##################################################

# ## Train and predict solar intensity using feature from satellite image and weather data
#Spli data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_images_weather, y_intensity, test_size=0.2, random_state=111)

#### Train different models.
# To learn how to preform training of such models as RNN networks containing time series we used
#https://www.tensorflow.org/tutorials/structured_data/time_series
# as reference


########### Model 1
model = keras.models.Sequential([
    keras.layers.LSTM(64,activation = "tanh",
                      return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(32,activation = "tanh",
                  return_sequences=True),
    keras.layers.Dense(units=1)
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=50,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000, callbacks=[early_stopping])


# In[ ]:


model.save('savedModel_LSTM_1')


########### Model 2
model = keras.models.Sequential([
    keras.layers.LSTM(64,activation = "tanh",
                      return_sequences=True),
    keras.layers.Dense(units=32),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(32,activation = "tanh",
                  return_sequences=True),
    keras.layers.Dense(units=1)
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=50,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000, callbacks=[early_stopping])


# In[ ]:


model.save('savedModel_LSTM_2')


########### Model 3

model = keras.models.Sequential([
    keras.layers.LSTM(64,activation = "relu",
                      return_sequences=True),
    keras.layers.LSTM(32,activation = "relu",
                  return_sequences=True),
    keras.layers.Dense(units=1)
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=50,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000, callbacks=[early_stopping])
model.save('savedModel_LSTM_3')


########### Model 4

model = keras.models.Sequential([
    keras.layers.LSTM(64,activation = "relu",
                      return_sequences=True),
    keras.layers.LSTM(64,activation = "relu",
                  return_sequences=True),
    keras.layers.Dense(units=1)
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=50,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000, callbacks=[early_stopping])
model.save('savedModel_LSTM_4')


########### Model 5

model = keras.models.Sequential([
    keras.layers.LSTM(64,activation = "relu",
                      return_sequences=True),
    keras.layers.Dense(units=1)
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=50,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000, callbacks=[early_stopping])
model.save('savedModel_LSTM_5')



########### Model 6

model = keras.models.Sequential([
    keras.layers.LSTM(64,activation = "tanh",
                      return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(32,activation = "tanh",
                  return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(32,activation = "tanh",
                  return_sequences=True),
    keras.layers.Dense(units=1)
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=50,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000, callbacks=[early_stopping])
model.save('savedModel_LSTM_6')
