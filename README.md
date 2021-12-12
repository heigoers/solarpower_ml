# solarpower_ml

This repository contains a source code and results of the group project done during the course Machine Learning (MTAT.03.227).

The main aim of the project is to be able to predict solar irradiance in near-future, using the weather data and satellite images.

The team consisted of:  Heigo Ers, Herman Klas Ratas and Mait Metelitsa

The repository consists following:

* 0_dataExploration.ipynb - contains preliminary data exploration and pair-wise correlations between the different weather data
* Approach 1: Predicting the solar irradiance of the next hour at a given location by using the weather data belonging to previous hour:
** 1a_estimateIntensityUsingWeatherData_all_variables.ipynb - Notebook testing different regressors with Min Max scaler preprocessing
** 1b_estimateIntensityUsingWeatherData_temp_time_rel_humidity.ipynb - Notebook testing different regressors with Min Max scaler, while using only chosen features
** 1c_estimateIntensityUsingWeatherData_all_variables_PCA.ipynb - Notebook testing different regressors, with PCA data preprocessing

* Approach 2: Predicting the solar irradiance of the next hour by using the weather data of other weather stations
** 2_automated_feature_selection_and model_blending.ipynb

* Approach 3: Predicting the solar irradiance of the next hour by using the timeseries, consiting of data extracted from satellite images
** 3_train_SolarIntensityUsingSatelliteData.py - Code, used for training the models on cluster
** 3_TrainingLog.out and 3_TrainingLog2.out - The logs of training
** 3_predict_SolarIntensityUsingSatelliteData.py - Code used for making the predictions using the trained models
** 3_predictionResults.out - The prediction results, for the trained models, with lowest mean absolute error during training
