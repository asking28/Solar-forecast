import netCDF4 as nc
from csv import reader
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy.random as r
import matplotlib.pyplot as plt
import pickle
import math
from numpy import random
from sklearn.utils import check_array
from pandas import read_csv
from pandas import datetime
precip=nc.Dataset( 'apcp_sfc_latlon_subset_19940101_20071231.nc' )['Total_precipitation'][:]
print(precip.shape)
precip=precip.reshape(5113,55,9,16)
precip=np.mean(precip,axis=1)
print(precip.shape)
precip=precip.reshape(5113,9*16)
print(precip.shape)
down_long=nc.Dataset('dlwrf_sfc_latlon_subset_19940101_20071231.nc')['Downward_Long-Wave_Rad_Flux'][:]
print(down_long.shape)
down_long=down_long.reshape(5113,55,9,16)
down_long=np.mean(down_long,axis=1)
print(down_long.shape)
down_long=down_long.reshape(5113,144)
print(down_long.shape)
down_short=nc.Dataset('dswrf_sfc_latlon_subset_19940101_20071231.nc')['Downward_Short-Wave_Rad_Flux'][:]
down_short=down_short.reshape(5113,55,9,16)
down_short=np.mean(down_short,axis=1)
down_short=down_short.reshape(5113,144)
print(down_short.shape)
pressure=nc.Dataset('pres_msl_latlon_subset_19940101_20071231.nc')['Pressure'][:]
pressure=pressure.reshape(5113,55,9,16)
pressure=np.mean(pressure,axis=1)
pressure=pressure.reshape(5113,144)
print(pressure.shape)
p_water=nc.Dataset('pwat_eatm_latlon_subset_19940101_20071231.nc')['Precipitable_water'][:]
p_water=p_water.reshape(5113,55,9,16)
p_water=np.mean(p_water,axis=1)
p_water=p_water.reshape(5113,144)
print(p_water.shape)
sp_humidity=nc.Dataset('spfh_2m_latlon_subset_19940101_20071231.nc')['Specific_humidity_height_above_ground'][:]
sp_humidity=sp_humidity.reshape(5113,55,9,16)
sp_humidity=np.mean(sp_humidity,axis=1)
sp_humidity=sp_humidity.reshape(5113,144)
print(sp_humidity.shape)
cloud_cover=nc.Dataset('tcdc_eatm_latlon_subset_19940101_20071231.nc')['Total_cloud_cover'][:]
cloud_cover=cloud_cover.reshape(5113,55,9,16)
cloud_cover=np.mean(cloud_cover,axis=1)
cloud_cover=cloud_cover.reshape(5113,144)
print(cloud_cover.shape)
int_condensate=nc.Dataset('tcolc_eatm_latlon_subset_19940101_20071231.nc')['Total_Column-Integrated_Condensate'][:]
int_condensate=int_condensate.reshape(5113,55,9,16)
int_condensate=np.mean(int_condensate,axis=1)
int_condensate=int_condensate.reshape(5113,144)
print(int_condensate.shape)
t_max=nc.Dataset('tmax_2m_latlon_subset_19940101_20071231.nc')['Maximum_temperature'][:]
t_max=t_max.reshape(5113,55,9,16)
t_max=np.mean(t_max,axis=1)
t_max=t_max.reshape(5113,144)
print(t_max.shape)
t_min=nc.Dataset('tmin_2m_latlon_subset_19940101_20071231.nc')['Minimum_temperature'][:]
t_min=t_min.reshape(5113,55,9,16)
t_min=np.mean(t_min,axis=1)
t_min=t_min.reshape(5113,144)
print(t_min.shape)
curr_temp=nc.Dataset('tmp_2m_latlon_subset_19940101_20071231.nc')['Temperature_height_above_ground'][:]
curr_temp=curr_temp.reshape(5113,55,9,16)
curr_temp=np.mean(curr_temp,axis=1)
curr_temp=curr_temp.reshape(5113,144)
print(curr_temp.shape)
surface_temp=nc.Dataset('tmp_sfc_latlon_subset_19940101_20071231.nc')['Temperature_surface'][:]
surface_temp=surface_temp.reshape(5113,55,9,16)
surface_temp=np.mean(surface_temp,axis=1)
surface_temp=surface_temp.reshape(5113,144)
print(surface_temp.shape)
up_long=nc.Dataset('ulwrf_sfc_latlon_subset_19940101_20071231.nc')['Upward_Long-Wave_Rad_Flux_surface'][:]
up_long=up_long.reshape(5113,55,9,16)
up_long=np.mean(up_long,axis=1)
up_long=up_long.reshape(5113,144)
print(up_long.shape)
up_top_long=nc.Dataset('ulwrf_tatm_latlon_subset_19940101_20071231.nc')['Upward_Long-Wave_Rad_Flux'][:]
up_top_long=up_top_long.reshape(5113,55,9,16)
up_top_long=np.mean(up_top_long,axis=1)
up_top_long=up_top_long.reshape(5113,144)
print(up_top_long.shape)
up_short=nc.Dataset('uswrf_sfc_latlon_subset_19940101_20071231.nc')['Upward_Short-Wave_Rad_Flux'][:]
up_short=up_short.reshape(5113,55,9,16)
up_short=np.mean(up_short,axis=1)
up_short=up_short.reshape(5113,144)
print(up_short.shape)
data=np.hstack((precip,down_long,down_short,pressure,p_water,sp_humidity,cloud_cover,int_condensate,t_max,t_min,curr_temp,surface_temp,up_long,up_top_long,up_short))
print(data.shape)
def parser(x):
	return datetime.strptime(x,'%Y%m%d')

dataset=read_csv('train.csv',parse_dates=[0],squeeze=True,date_parser=parser)
print(dataset.head(10))
print(dataset.shape)
dset=np.asarray(dataset)
print(dset.shape)
dates=dset[:,0]
Y=dset[:,1:]
print(dates.shape)
print(Y.shape)
#print(Y.head(10))
seed=7
X_train,X_test,Y_train,Y_test=train_test_split(data,Y[:,:],test_size=0.4,random_state=seed)
print(X_train.shape,Y_train.shape)
print(X_train)
max_abs_scaler=preprocessing.MaxAbsScaler()

X_train=preprocessing.scale(X_train)
print(X_train)
#mean 0 standard deviation 1
X_test=preprocessing.scale(X_test)
print(Y_train.shape[1])
nn_structure=[2160,1200,500,98]
Y_tr=Y_train.astype(np.float)
y_tst=Y_test.astype(np.float)
from sklearn.neural_network import MLPRegressor
mlp=MLPRegressor(hidden_layer_sizes=(1200,500,),activation='relu',solver='adam',alpha=0.0001,max_iter=500)
mlp.fit(X_train,Y_tr)
preds=mlp.predict(X_test)
mae=metrics.mean_absolute_error(y_tst,preds)
r2_scre=metrics.r2_score(y_tst,preds)
from sklearn.utils import check_array
def mean_absolute_percentage_error(y_tst,preds):
	y_tst=check_array(y_tst)
	preds=check_array(preds)
	return np.mean(np.abs((y_tst-preds)/y_tst))*100.0
mape=mean_absolute_percentage_error(y_tst,preds)
print(mae)
print(r2_scre)

