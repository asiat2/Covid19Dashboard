#======= Visualization, analysis and machine learning model for california housing dataset
#==========================================================================================
#============== Ashiat Adeogun (january 2021)
#=========================================================================================
#===================================================================================
#import required libraries
#===================================================================================
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
# ====================Load california housing dataset==============================
df_califonia =pd.read_csv('housing.csv',sep=',')

#====================Data visualization ===========================================
fig, axes = plt.subplots(nrows=3, ncols=3,sharex=False,sharey=False,gridspec_kw=None)   # create 3x3 array of subplots
fig.subplots_adjust(wspace=0.5,hspace=0.5)
df_califonia.boxplot(column='latitude', ax=axes[0,0]) 
df_califonia.boxplot(column='longitude', ax=axes[0,1])
df_califonia.boxplot(column='total_rooms', ax=axes[0,2]) 
df_califonia.boxplot(column='total_bedrooms', ax=axes[1,0]) 
df_califonia.boxplot(column='population', ax=axes[1,1])
df_califonia.boxplot(column='households', ax=axes[1,2])
df_califonia.boxplot(column='median_income', ax=axes[2,0])
df_califonia.boxplot(column='housing_median_age', ax=axes[2,1])
df_califonia.boxplot(column='median_house_value', ax=axes[2,2])
plt.show()

# plotting correlation heatmap 
corr=df_califonia.corr()
dataplot = sns.heatmap(np.abs(corr), cmap="YlGnBu", annot=True) 
plt.show()

# Creating the scatter matrix
ax=pd.plotting.scatter_matrix(df_califonia.iloc[:,2:9],ax=None,alpha=0.5,figsize=(12,12),diagonal='hist')

# ================ Data preprocessing =============================================
df_califonia.info
df_califonia.describe()
df_califonia.isna().sum()
df_califonia.dropna(how='any',inplace=True)
df_califonia.duplicated().sum()

#checking for outliers
Q1 = df_califonia.quantile(0.25)
Q3 = df_califonia.quantile(0.75)
IQR = Q1 - Q3
print(Q1)
print(Q3)
print(IQR)
lower_bound= Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
print(upper_bound)
print(lower_bound)
print(((df_califonia<(lower_bound))|(df_califonia >(upper_bound))).sum())

df_califonia.shape

# ================Build ML regression model ======================================= 

df_select = df_califonia.loc[:,['total_rooms','total_bedrooms','households']]


#================Prepare data for ML training ===================================
df_train, df_test = train_test_split(df_califonia,test_size = 0.3)

print('predict model for population')
#================Training ML model ===============================================
x_train=df_train.loc[:,['total_rooms','total_bedrooms','households']]
y_train =df_train.loc[:,['population']]
x_test=df_test.loc[:,['total_rooms','total_bedrooms','households']]
y_test =df_test.loc[:,['population']]
#==== Multi-Layer Perceptron (MLP) Model
nn_model = MLPRegressor(hidden_layer_sizes=(200,100),max_iter=5000)
nn_model.fit(X=x_train,y=y_train)
y_test_predicted_nn = nn_model.predict(x_test)
report_nn = nn_model.score(x_test,y_test)
print("Multi-Layer Perceptron")
print(report_nn)
#=======Muliple Linear Regression====================================
lr_model = LinearRegression()
lr_model.fit(X=x_train,y=y_train)
y_test_predicted = lr_model.predict(x_test)
report_lr = lr_model.score(x_test,y_test)
print("Linear_Regression")
print(report_lr)
#========================decisiontree=========================
dt_model = DecisionTreeRegressor(criterion='mse',splitter='best',)
dt_model.fit(X=x_train,y=y_train)
y_test_predicted_dt = dt_model.predict(x_test)
report_dt = dt_model.score(x_test,y_test)
print("DecisionTreeRegressor")
print(report_dt)

print('predict model for households')
#==============Training ML Model for households prediction============================
x_train=df_train.loc[:,['total_rooms','total_bedrooms','population']]
y_train =df_train.loc[:,['households']]
x_test=df_test.loc[:,['total_rooms','total_bedrooms','population']]
y_test =df_test.loc[:,['households']]
#==== Multi-Layer Perceptron (MLP) Model households====================================
nn_model = MLPRegressor(hidden_layer_sizes=(200,100),max_iter=5000)
nn_model.fit(X=x_train,y=y_train)
y_test_predicted_nn = nn_model.predict(x_test)
report_nn = nn_model.score(x_test,y_test)
print("Multi-Layer Perceptron")
print(report_nn)
#=======Muliple Linear Regression households====================================
lr_model = LinearRegression()
lr_model.fit(X=x_train,y=y_train)
y_test_predicted = lr_model.predict(x_test)
report_lr = lr_model.score(x_test,y_test)
print("Linear_Regression")
print(report_lr)
#========================decisiontree households=========================
dt_model = DecisionTreeRegressor(criterion='mse',splitter='best',)
dt_model.fit(X=x_train,y=y_train)
y_test_predicted_dt = dt_model.predict(x_test)
report_dt = dt_model.score(x_test,y_test)
print("DecisionTreeRegressor")
print(report_dt)

print('predict model for total_rooms')
#==============Training ML Model for total_rooms prediction============================
x_train=df_train.loc[:,['households','total_bedrooms','population']]
y_train =df_train.loc[:,['total_rooms']]
x_test=df_test.loc[:,['households','total_bedrooms','population']]
y_test =df_test.loc[:,['total_rooms']]
#==== Multi-Layer Perceptron (MLP) Model households====================================
nn_model = MLPRegressor(hidden_layer_sizes=(200,100),max_iter=5000)
nn_model.fit(X=x_train,y=y_train)
y_test_predicted_nn = nn_model.predict(x_test)
report_nn = nn_model.score(x_test,y_test)
print("Multi-Layer Perceptron")
print(report_nn)
#=======Muliple Linear Regression Model total_rooms====================================
lr_model = LinearRegression()
lr_model.fit(X=x_train,y=y_train)
y_test_predicted = lr_model.predict(x_test)
report_lr = lr_model.score(x_test,y_test)
print("Linear_Regression")
print(report_lr)
#========================Decision Tree Model total_rooms=========================
dt_model = DecisionTreeRegressor(criterion='mse',splitter='best',)
dt_model.fit(X=x_train,y=y_train)
y_test_predicted_dt = dt_model.predict(x_test)
report_dt = dt_model.score(x_test,y_test)
print("DecisionTreeRegressor")
print(report_dt)


print('predict model for total_bedrooms')
#==============Training ML Model for totals_bedroom prediction============================
x_train=df_train.loc[:,['households','total_rooms','population']]
y_train =df_train.loc[:,['total_bedrooms']]
x_test=df_test.loc[:,['households','total_rooms','population']]
y_test =df_test.loc[:,['total_bedrooms']]
#==== Multi-Layer Perceptron (MLP) Model total_bedrooms====================================
nn_model = MLPRegressor(hidden_layer_sizes=(200,100),max_iter=5000)
nn_model.fit(X=x_train,y=y_train)
y_test_predicted_nn = nn_model.predict(x_test)
report_nn = nn_model.score(x_test,y_test)
print("Multi-Layer Perceptron")
print(report_nn)
#=======Muliple Linear Regression Model total_bedrooms====================================
lr_model = LinearRegression()
lr_model.fit(X=x_train,y=y_train)
y_test_predicted = lr_model.predict(x_test)
report_lr = lr_model.score(x_test,y_test)
print("Linear_Regression")
print(report_lr)
#========================Decision Tree model for total_bedrooms=========================
dt_model = DecisionTreeRegressor(criterion='mse',splitter='best',)
dt_model.fit(X=x_train,y=y_train)
y_test_predicted_dt = dt_model.predict(x_test)
report_dt = dt_model.score(x_test,y_test)
print("DecisionTreeRegressor")
print(report_dt)