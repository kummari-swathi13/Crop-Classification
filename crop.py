import pandas as pd
#pd is universal standard

crop_data=pd.read_csv(r"C:\Users\kumma\OneDrive\Documents\crop.csv")
crop_data.shape #shape is an attribute

#displays the top records
crop_data.head(10)

#displays the bottom records
crop_data.tail(10)

#displays the data set information
crop_data.info()

#displays the discriptive statistics
#describe functions only works for non-numeric
crop_data.describe()
print("mean")
crop_data.N.mean()
crop_data.P.mean()
crop_data.K.mean()
crop_data.temperature.mean()
crop_data.humidity.mean()
crop_data.ph.mean()
crop_data.rainfall.mean()


print("median")
crop_data.N.median()
crop_data.P.median()
crop_data.K.median()
crop_data.temperature.median()
crop_data.humidity.median()
crop_data.ph.median()
crop_data.rainfall.median()

print("mode")
crop_data.N.mode()
crop_data.P.mode()
crop_data.K.mode()
crop_data.temperature.mode()
crop_data.humidity.mode()
crop_data.ph.mode()
crop_data.rainfall.mode()
crop_data.label.mode()

#second moment business decision i.e dispersion from center
#used to find measures
print("variance")
crop_data.N.var()#variance of the column
crop_data.P.var()
crop_data.K.var()
crop_data.temperature.var()
crop_data.humidity.var()
crop_data.ph.var()
crop_data.rainfall.var()
#crop_data.label.var()

print("std")
crop_data.N.std()
crop_data.P.std()
crop_data.K.std()
crop_data.temperature.std()
crop_data.humidity.std()
crop_data.ph.std()
crop_data.rainfall.std()
#crop_data.label.var()

print("range")
max(crop_data.N)-min(crop_data.K)
max(crop_data.N)-min(crop_data.P)
max(crop_data.temperature)-min(crop_data.humidity)
max(crop_data.ph)-min(crop_data.rainfall)
max(crop_data.rainfall)-min(crop_data.humidity)
max(crop_data.temperature)-min(crop_data.ph)


#3rd moment business decison i.e it talk about directions of the outliers either left or right
#used to find skewness i.e +skew and -skew
#we also have relationship b/w 1st and 2nd moment
#we can't calculate skewness foe non negitive column
#for +skew mean>median>mode 
crop_data.N.skew()
crop_data.K.skew()
crop_data.P.skew()
crop_data.temperature.skew()
crop_data.humidity.skew()
crop_data.ph.skew()
crop_data.rainfall.skew()

#4th moment business decision 
crop_data.N.kurt()#positive kurt(means sharp peak) and negative kurt() means wide peak
crop_data.K.kurt()
crop_data.P.kurt()
crop_data.temperature.kurt()
crop_data.humidity.kurt()#positive kurt(means sharp peak) and negative kurt() means wide peak
crop_data.ph.kurt()
crop_data.rainfall.kurt()


#DATA VISUALIZATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # pyplot is used for mrmory optimization
import seaborn as sns # it is an advanced visualization of matplotlib #pip means package installer for python

crop_data=pd.read_csv(r"C:\Users\kumma\OneDrive\Documents\crop.csv")

crop_data.info()

plt.bar(height=crop_data.N,x=range(2200))
plt.show()

plt.bar(height=crop_data.P,x=range(2200))
plt.show()

plt.bar(height=crop_data.temperature,x=range(2200))
plt.show()

plt.bar(height=crop_data.humidity,x=range(2200))
plt.show()

plt.bar(height=crop_data.ph,x=range(2200))
plt.show()

plt.bar(height=crop_data.rainfall,x=range(2200))
plt.show()

#histogram is for data distribution
plt.hist(crop_data.N)
plt.title('histogram of nitrogen')
plt.show()

plt.hist(crop_data.P)
plt.title('histogram of Phosphorous content in soil')
plt.show()

plt.hist(crop_data.K)
plt.title('histogram of Potassium content in soil')
plt.show()

plt.hist(crop_data.temperature)
plt.title('histogram of Tempertaure')
plt.show()

plt.hist(crop_data.humidity)
plt.title('histogram of humidity')
plt.show()

plt.hist(crop_data.ph)
plt.title('histogram of ph value in soil')
plt.show()

plt.hist(crop_data.rainfall)
plt.title('histogram of Rainfall in mm')
plt.show()

#plt.hist(crop_data.label)
#plt.show()
plt.figure(figsize=(8,5))
plt.hist(crop_data.N,
         bins=[4,5,6,7,8],
         color='red',
         edgecolor='black')
plt.xlabel('Nitrogen content in soil')
plt.ylabel('frequency')
plt.title('histogram of nitrogen')
plt.show()

plt.figure(figsize=(8,5))
plt.hist(crop_data.P,
         bins=[4,5,6,7,8],
         color='red',
         edgecolor='black')
plt.xlabel('Phospherus')
plt.ylabel('frequency')
plt.title('histogram of phospherus content in soil')
plt.show()

#boxplot is used to visualize outliers
#we cannot tell no of outliers in box plot
plt.boxplot(crop_data.P)
plt.title('boxplot of Phosphorous')
plt.show()

plt.boxplot(crop_data.K)
plt.title('Boxplot of Potassium')
plt.show()

plt.boxplot(crop_data.temperature)
plt.title('Boxplot of Temperature')
plt.show()

plt.boxplot(crop_data.humidity)
plt.title('Boxplot of humidity')
plt.show()

plt.boxplot(crop_data.ph)
plt.title('Boxplot of Ph value of soil')
plt.show()

plt.boxplot(crop_data.rainfall)
plt.title('Boxplot of Rainfall in mm')
plt.show()

plt.boxplot(crop_data.label)
plt.title('Boxplot of nitrogen')
plt.show()

sns.boxplot(crop_data.humidity)
plt.title('seabornn of nitrogen')# seaborn is advanced visualization
plt.show()

#scatter plot is used for understanding the relationship b/w two columns or values
#scatter plots have clustes i.e groups
#co-relation coefficients
#r>0.85---->strong
#0.85<r<0.4----->moderate
#r<0.85---->weak
plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.N,y=crop_data.K,color='green')
plt.xlabel('Nitrogen content in soil')
plt.ylabel('potassium content in soil')
plt.title('scatter of Nitrogen and potassium')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()

plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.N,y=crop_data.P,color='green')
plt.xlabel('Nitrogen content in soil')
plt.ylabel('phosphorous content in soil')
plt.title('scatter of Nitrogen and phosphorus')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()


plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.K,y=crop_data.P,color='green')
plt.xlabel('Potassium content in soil')
plt.ylabel('phosphorous content in soil')
plt.title('scatter of potassium and phosphorus')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()

plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.temperature,y=crop_data.humidity,color='blue')
plt.xlabel('Temperature in degree C')
plt.ylabel('Relative humidity in %')
plt.title('scatter of temperataure and Humidity')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()

plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.humidity,y=crop_data.ph,color='red')
plt.xlabel('Relative humidity in %')
plt.ylabel('ph value of the soil')
plt.title('scatter of Humidity and ph value')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()

plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.ph,y=crop_data.rainfall,color='green')
plt.xlabel('ph value of the soil')
plt.ylabel('Rainfall in mm')
plt.title('scatter of Ph value and Rainfall')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()

plt.figure(figsize=(10,8))
plt.scatter(x=crop_data.P,y=crop_data.temperature,color='orange')
plt.xlabel('Potassium contentb in soil')
plt.ylabel('Temperature in degree')
plt.title('scatter of Phosphorous and Temperature')
plt.show()
#iloc---->indexing location
crop_data.iloc[:,0:4].corr()



#WINSORIZER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
crops=pd.read_csv(r"C:\Users\kumma\OneDrive\Documents\crop.csv")
crops.info
crops.plot(kind='box',
        figsize=(12,8),
        subplots=True,
        sharey=False,
        color={"boxes":"blue","whiskers":"red","medians":"pink"},
        patch_artist=True,
        showfliers=True)
plt.show()
from feature_engine.outliers import Winsorizer
winsor_iqr=Winsorizer(capping_method='iqr',
                      tail='both',
                      fold=1.5,
                      variables=['N'])
winsor_iqr2=Winsorizer(capping_method='iqr',
                      tail='both',
                      fold=1.5,
                      variables=['P'])



#missing values
import numpy as np
import pandas as pd
crop=pd.read_csv(r"C:\Users\kumma\OneDrive\Documents\crop.csv")
crop.info()
crop.shape
print("\nCount of missing values")
print(crop.isna().sum())
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["N"]=mean_imputer.fit_transform(crop[["N"]])
crop["N"].isna().sum()

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["P"]=mean_imputer.fit_transform(crop[["P"]])
crop["P"].isna().sum()

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["K"]=mean_imputer.fit_transform(crop[["K"]])
crop["K"].isna().sum()

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["temperature"]=mean_imputer.fit_transform(crop[["temperature"]])
crop["temperature"].isna().sum()

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["humidity"]=mean_imputer.fit_transform(crop[["humidity"]])
crop["humidity"].isna().sum()

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["ph"]=mean_imputer.fit_transform(crop[["ph"]])
crop["ph"].isna().sum()

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
crop["rainfall"]=mean_imputer.fit_transform(crop[["rainfall"]])
crop["rainfall"].isna().sum()



mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
crop["label"]=mode_imputer.fit_transform(crop[["label"]])[:,0]
crop["label"].isnull().sum()
print(crop)

#DUPLICATES
import pandas as pd

crop_data=pd.read_csv(r"C:\Users\kumma\OneDrive\Documents\crop.csv")

duplicate_first = crop_data.duplicated(keep='first')
duplicate_last = crop_data.duplicated(keep='last')
duplicate_all = crop_data.duplicated(keep=False)

# Print out boolean series for duplicate rows
print("Boolean series indicating duplicate rows (keeping first occurrence):")
print(duplicate_first)

# Count total duplicates (based on 'first' occurrence)
total_duplicate_first = sum(duplicate_first)
print(f"\nTotal number of duplicate rows (keeping first occurrence): {total_duplicate_first}")

# Count total duplicates (based on 'last' occurrence)
total_duplicate_last = sum(duplicate_last)
print(f"\nTotal number of duplicate rows (keeping last occurrence): {total_duplicate_last}")

# Count all duplicates, ignoring the occurrence order
total_duplicate_all = sum(duplicate_all)
print(f"\nTotal number of duplicate rows (keeping neither first nor last): {total_duplicate_all}")

# Remove duplicates, keeping the first occurrence
data_first = crop_data.drop_duplicates(keep='first')
print(f"\nShape of the data after removing duplicates (keeping first occurrence): {data_first.shape}")

# Remove duplicates, keeping the last occurrence
data_last = crop_data.drop_duplicates(keep='last')
print(f"Shape of the data after removing duplicates (keeping last occurrence): {data_last.shape}")

# Remove all duplicates, keeping no occurrences
data_all = crop_data.drop_duplicates(keep=False)
print(f"Shape of the data after removing all duplicates: {data_all.shape}")

import pandas as pd
data=pd.read_csv(r"C:\Users\kumma\OneDrive\Documents\crop.csv")
des=data.describe()
data1=data.iloc[:,0:7]

#data normalization-> minmax scaler
from sklearn.preprocessing import MinMaxScaler

#initializing the MinMaxScaler
minmaxscaler=MinMaxScaler()

#scaling data using MinMaxScaler
crop_norm=minmaxscaler.fit_transform(data1)#it is an array

#convert scaled array back to dataframe
dataset_norm=pd.DataFrame(crop_norm)
res=dataset_norm.describe()


