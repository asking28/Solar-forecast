import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import netCDF4 as nc
from csv import reader
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy.random as r
import matplotlib.pyplot as plt
import pickle
precip=nc.Dataset( 'apcp_sfc_latlon_subset_19940101_20071231.nc' )['Total_precipitation'][:]
print(precip.shape)
precip=precip.reshape(5113,55,9,16)
precip=np.mean(precip,axis=1)
print(precip.shape)
precip=precip.reshape(5113,9*16)
column=['precipcol_'+str(i) for i in range(1,145)]
precip=pd.DataFrame(precip,columns=column)

print(precip.head(10))
down_long=nc.Dataset('dlwrf_sfc_latlon_subset_19940101_20071231.nc')['Downward_Long-Wave_Rad_Flux'][:]
print(down_long.shape)
down_long=down_long.reshape(5113,55,9,16)
down_long=np.mean(down_long,axis=1)
print(down_long.shape)
down_long=down_long.reshape(5113,144)
column1=['longdown_'+str(i) for i in range(1,145)]
down_long=pd.DataFrame(down_long,columns=column1)
column2=np.append(column,column1)
#print(column2)
#print(down_long.shape)
down_short=nc.Dataset('dswrf_sfc_latlon_subset_19940101_20071231.nc')['Downward_Short-Wave_Rad_Flux'][:]
down_short=down_short.reshape(5113,55,9,16)
down_short=np.mean(down_short,axis=1)
down_short=down_short.reshape(5113,144)
column1=['down_short_'+str(i) for i in range(1,145)]
down_short=pd.DataFrame(down_short,columns=column1)
column2=np.append(column2,column1)
#print(down_short.shape)
pressure=nc.Dataset('pres_msl_latlon_subset_19940101_20071231.nc')['Pressure'][:]
pressure=pressure.reshape(5113,55,9,16)
pressure=np.mean(pressure,axis=1)
pressure=pressure.reshape(5113,144)
column1=['pressure_'+str(i) for i in range(1,145)]
pressure=pd.DataFrame(pressure,columns=column1)
column2=np.append(column2,column1)
#print(pressure.shape)
p_water=nc.Dataset('pwat_eatm_latlon_subset_19940101_20071231.nc')['Precipitable_water'][:]
p_water=p_water.reshape(5113,55,9,16)
p_water=np.mean(p_water,axis=1)
p_water=p_water.reshape(5113,144)
column1=['pwater_'+str(i) for i in range(1,145)]
p_water=pd.DataFrame(p_water,columns=column1)
column2=np.append(column2,column1)
#print(p_water.shape)
sp_humidity=nc.Dataset('spfh_2m_latlon_subset_19940101_20071231.nc')['Specific_humidity_height_above_ground'][:]
sp_humidity=sp_humidity.reshape(5113,55,9,16)
sp_humidity=np.mean(sp_humidity,axis=1)
sp_humidity=sp_humidity.reshape(5113,144)
column1=['sphumidity_'+str(i) for i in range(1,145)]
sp_humidity=pd.DataFrame(sp_humidity,columns=column1)
column2=np.append(column2,column1)
#print(sp_humidity.shape)
cloud_cover=nc.Dataset('tcdc_eatm_latlon_subset_19940101_20071231.nc')['Total_cloud_cover'][:]
cloud_cover=cloud_cover.reshape(5113,55,9,16)
cloud_cover=np.mean(cloud_cover,axis=1)
cloud_cover=cloud_cover.reshape(5113,144)
column1=['cloudcover_'+str(i) for i in range(1,145)]
cloud_cover=pd.DataFrame(cloud_cover,columns=column1)
column2=np.append(column2,column1)
#print(cloud_cover.shape)
int_condensate=nc.Dataset('tcolc_eatm_latlon_subset_19940101_20071231.nc')['Total_Column-Integrated_Condensate'][:]
int_condensate=int_condensate.reshape(5113,55,9,16)
int_condensate=np.mean(int_condensate,axis=1)
int_condensate=int_condensate.reshape(5113,144)
column1=['condensate_'+str(i) for i in range(1,145)]
int_condensate=pd.DataFrame(int_condensate,columns=column1)
column2=np.append(column2,column1)
#print(int_condensate.shape)
t_max=nc.Dataset('tmax_2m_latlon_subset_19940101_20071231.nc')['Maximum_temperature'][:]
t_max=t_max.reshape(5113,55,9,16)
t_max=np.mean(t_max,axis=1)
t_max=t_max.reshape(5113,144)
column1=['tmax_'+str(i) for i in range(1,145)]
t_max=pd.DataFrame(t_max,columns=column1)
column2=np.append(column2,column1)
#print(t_max.shape)
t_min=nc.Dataset('tmin_2m_latlon_subset_19940101_20071231.nc')['Minimum_temperature'][:]
t_min=t_min.reshape(5113,55,9,16)
t_min=np.mean(t_min,axis=1)
t_min=t_min.reshape(5113,144)
column1=['tmin_'+str(i) for i in range(1,145)]
t_min=pd.DataFrame(t_min,columns=column1)
column2=np.append(column2,column1)
#print(t_min.shape)
curr_temp=nc.Dataset('tmp_2m_latlon_subset_19940101_20071231.nc')['Temperature_height_above_ground'][:]
curr_temp=curr_temp.reshape(5113,55,9,16)
curr_temp=np.mean(curr_temp,axis=1)
curr_temp=curr_temp.reshape(5113,144)
column1=['currtemp_'+str(i) for i in range(1,145)]
curr_temp=pd.DataFrame(curr_temp,columns=column1)
column2=np.append(column2,column1)
#print(curr_temp.shape)"""
surface_temp=nc.Dataset('tmp_sfc_latlon_subset_19940101_20071231.nc')['Temperature_surface'][:]
surface_temp=surface_temp.reshape(5113,55,9,16)
surface_temp=np.mean(surface_temp,axis=1)
surface_temp=surface_temp.reshape(5113,144)
column1=['surfacetemp_'+str(i) for i in range(1,145)]
surface_temp=pd.DataFrame(surface_temp,columns=column1)
column2=np.append(column2,column1)

#print(surface_temp.shape)
up_long=nc.Dataset('ulwrf_sfc_latlon_subset_19940101_20071231.nc')['Upward_Long-Wave_Rad_Flux_surface'][:]
up_long=up_long.reshape(5113,55,9,16)
up_long=np.mean(up_long,axis=1)
up_long=up_long.reshape(5113,144)
column1=['uplong_'+str(i) for i in range(1,145)]
up_long=pd.DataFrame(up_long,columns=column1)
column2=np.append(column2,column1)
#print(up_long.shape)
up_top_long=nc.Dataset('ulwrf_tatm_latlon_subset_19940101_20071231.nc')['Upward_Long-Wave_Rad_Flux'][:]
up_top_long=up_top_long.reshape(5113,55,9,16)
up_top_long=np.mean(up_top_long,axis=1)
up_top_long=up_top_long.reshape(5113,144)
column1=['uptop_'+str(i) for i in range(1,145)]
up_top_long=pd.DataFrame(up_top_long,columns=column1)
column2=np.append(column2,column1)
#print(up_top_long.shape)
up_short=nc.Dataset('uswrf_sfc_latlon_subset_19940101_20071231.nc')['Upward_Short-Wave_Rad_Flux'][:]
up_short=up_short.reshape(5113,55,9,16)
up_short=np.mean(up_short,axis=1)
up_short=up_short.reshape(5113,144)
column1=['upshort_'+str(i) for i in range(1,145)]
up_short=pd.DataFrame(up_short,columns=column1)
column2=np.append(column2,column1)
print(up_short.shape)


data=np.hstack((precip,down_long,down_short,pressure,p_water,sp_humidity,cloud_cover,int_condensate,t_max,t_min,curr_temp,surface_temp,up_long,up_top_long,up_short))
data=pd.DataFrame(data,columns=column2)
#print(data.head(3))
def load_output(filename):
	dataset=list()
	with open(filename,'r')as file:
		csv_reader=reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
			
	return dataset
	
dataset=load_output('train.csv')
dset=np.asarray(dataset)
Y=dset[1:,1:]
Y=Y.astype(np.float)
def baseline_model():
	model=Sequential()	
	model.add(Dense(1500,input_dim=2160,activation='relu',kernel_initializer='normal'))
	model.add(Dense(1200,activation='relu',kernel_initializer='normal'))
	model.add(Dense(900,activation='relu',kernel_initializer='normal'))
	model.add(Dense(700,activation='relu',kernel_initializer='normal'))
	model.add(Dense(400,activation='relu',kernel_initializer='normal'))
	model.add(Dense(200,activation='relu',kernel_initializer='normal'))
	model.add(Dense(100,activation='relu',kernel_initializer='normal'))
	model.add(Dense(50,activation='relu',kernel_initializer='normal'))
	model.add(Dense(10,activation='relu',kernel_initializer='normal'))
	
	model.add(Dense(98,kernel_initializer='normal'))
	model.compile(loss='mean_squared_error',optimizer='adam')
	return model
seed=7
data=data.as_matrix()
scale=StandardScaler()
data=scale.fit_transform(data)
np.random.seed(seed)
print(data.shape)
estimator=KerasRegressor(build_fn=baseline_model,nb_epoch=3000,batch_size=5,verbose=0)
X_train,X_test,Y_train,Y_test=train_test_split(data,Y,test_size=0.20,random_state=seed)
print(X_train.shape)
print(Y_train.shape)

estimator.fit(X_train,Y_train)
score=metrics.mean_absolute_error(Y_test,estimator.predict(X_test))
print(score)
