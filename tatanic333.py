#======= Visualization, analysis and machine learning model for titanic dataset
#==========================================================================================
#============== Ashiat Adeogun (december 2020)
#=========================================================================================
#===================================================================================
#import required libraries
#===================================================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# ====================Load titanic dataset==============================
df_tatanic =pd.read_csv('titanic.csv',sep=',',index_col='PassengerId')

#======================function processing==============================
#Fillna for Age with average number of Pclass and sex
average_ages = df_tatanic.groupby(['Pclass','Sex']).mean().Age.astype(int).to_dict()
average_ages
def age_guesser(pclass_sex):
    Pclass_sex=tuple(pclass_sex)
    age =average_ages[Pclass_sex]
    return age
raw_df_tatanic = df_tatanic
for pclass,sex in average_ages.keys():
    df_tatanic.loc[(df_tatanic.Pclass==pclass) & (df_tatanic.Sex == sex),"Age"]=(
    df_tatanic.loc[(df_tatanic.Pclass==pclass) & (df_tatanic.Sex == sex),"Age"].fillna(age_guesser([pclass,sex])))
    
#craete a empty list for class to change the pclass column from numerical to categorical
def pclass(x):
    for i in range(1,len(x['Pclass'])+1):
        if x['Pclass'][i] == 1:
             x['Pclass'][i] = 'First'
        elif x['Pclass'][i] == 2:
            x['Pclass'][i]='Second'
        else:
            x['Pclass'][i] = 'Third'
    return x

#Function to split the column Name
def split_name(x):
    df_tatanic[['last_title','first_name']] = (df_tatanic['Name'].str.split(('.'),n=1,expand=True))
    df_tatanic[['last_name','title']] = (df_tatanic['last_title'].str.split(',',n=1,expand=True))
    df_tatanic.drop(columns='last_title', inplace=True)
    df_tatanic['title']=df_tatanic['title'].str.replace(' ','')
    return x 

#Changing the Embarked column to full word
def embark(x):
    for i in range(1,len(x['Embarked'])+1):
        if x['Embarked'][i] in ['S']:
             x['Embarked'][i] = 'Sounthampton'
        elif x['Embarked'][i] in ['C']:
             x['Embarked'][i]='Cherbourg'
        else:
            x['Embarked'][i] = 'Queenstown'
    return x

#Function for grouping the Age
def age_group(x):
    Age_group=[]
    age =x['Age']
    for i in range(1,len(age)+1):
        if age[i]<=3:
             age_group = 'Toddler'
        elif age[i]>3 and age[i]<=5:
             age_group ='Preschool'
        elif age[i]>5 and age[i]<=12: 
             age_group='Gradeschool'
        elif age[i]>12 and age[i]<=19: 
             age_group ='Teen'
        elif age[i]>19 and age[i]<=35: 
             age_group ='YoungAdult'
        elif age[i]>35 and age[i]<=60:
             age_group ='MiddleAged'
        else:
             age_group='Senior'
        Age_group.append(age_group)
    x['Age_group'] =  Age_group
    return x

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    tr = split_name(x)
    tr=pd.DataFrame(tr)
    title=tr['title']
    sex =tr['Sex']
    for i in range(1,len(title)+1):
        if title[i] in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:
            title[i] = 'Mr'
        elif title[i]  in ['theCountess', 'Mme']:
            title[i] = 'Mrs'
        elif title[i]  in ['Mlle', 'Ms','Lady']:
            title[i] = 'Miss'
        elif title[i] =='Dr':
            if sex[i] =='male':
                title[i] ='Mr'
            else:
                title[i] ='Mrs'
        x['title'][i] = title[i]
    return x

#Function to edit columns
def edit_colum(x):
    df_tatanic['Sex'] = df_tatanic['Sex'].str.title()#modify Sex column male and female by changing the first letters into capital letter
    df_tatanic['Survived'] = df_tatanic['Survived'].replace({1:'Yes',0:'No'})
    return x
#Calling all the functions
def final_process(x):
    df_1=pclass(x)
    df_2=split_name(df_1)
    df_3=embark(df_2)
    df_4=age_group(df_3)
    df_5=replace_titles(df_4)
    df_6=edit_colum(df_5)
    return df_6

#function to  plot visualize categorical variable
def plot_bar(column,x):
    # temp df 
    temp_1 = pd.DataFrame()#empty df
    # count categorical values
    temp_1['Yes_Survied'] = df_tatanic[df_tatanic['Survived'] == 'Yes'][column].value_counts()
    temp_1['No_Survived'] = df_tatanic[df_tatanic['Survived'] == 'No'][column].value_counts()
    temp_1.plot(kind ='bar')
    plt.xlabel(f'{column}')
    plt.ylabel('Number of survived')
    plt.title('Distribution of {} and Survived'.format(column))
    plt.show()
    
# ================ Data preprocessing =============================================
#Reduce the memory usage
df_tatanic_final = final_process(df_tatanic)  
df_tatanic_final.info()
df_tatanic_final=df_tatanic_final.drop(['Cabin'],axis=1)#drop Cabin column
df_tatanic_final['Sex'] = df_tatanic_final['Sex'].astype('category')
df_tatanic_final['Embarked'] = df_tatanic_final['Embarked'].astype('category')
df_tatanic_final['Age_group'] = df_tatanic_final['Age_group'].astype('category')
df_tatanic_final['Pclass'] = df_tatanic_final['Pclass'].astype('category')
df_tatanic_final['title'] = df_tatanic_final['title'].astype('category')


#====================Data visualization  ===========================================
#Visualize the categorical varibles using barchat
plot_bar('Sex',df_tatanic_final)
plot_bar('Pclass',df_tatanic_final)
plot_bar('Embarked',df_tatanic_final)
plot_bar('Age_group',df_tatanic_final)
plot_bar('title',df_tatanic_final)

#plot piechat for Sex and to know the percentage
df_tatanic_final['Sex'].value_counts()
labels = 'Female' , 'Male'
sizes = [ 314,577]
colors = ['pink', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140,  frame = True,wedgeprops   = { 'linewidth' : 3,'edgecolor' : "yellowgreen"})
plt.axis('equal')
plt.show()

#pie plot to know the percentages of each classes in the tatanic
df_tatanic_final['Pclass'].value_counts()
labels = 'First' , 'Second','Third'
sizes = [ 216,184,491]
colors = ['pink', 'yellowgreen','blue']
explode = (0.1,0,0.1)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140,  frame = True,wedgeprops   = { 'linewidth' : 3,'edgecolor' : "yellowgreen"})
plt.axis('equal')
plt.show()

#===========visualisation=====================================================
df_processed=df_tatanic_final
fig, axes = plt.subplots(nrows=2, ncols=2,sharex=False,sharey=False,gridspec_kw=None)   # create 3x3 array of subplots
fig.subplots_adjust(wspace=0.5,hspace=0.5)
df_processed.boxplot(column='Fare', ax=axes[0,0]) 
df_processed.boxplot(column='SibSp', ax=axes[0,1])
df_processed.boxplot(column='Age', ax=axes[1,0])
df_processed.boxplot(column='Parch', ax=axes[1,1]) 
plt.show()

#=============processing=============================================================
#processing for ML 
df_processed =df_tatanic_final.drop(columns=["Name","Ticket","Embarked","Age_group",
                            "title","last_name","first_name"]) #drop the columns
df_processed['Pclass'] = df_processed['Pclass'].replace({'First':1,'Second':2,'Third':3}) #replace the Pclass to numeric variable
df_processed.loc[:,["Survived"]] = pd.get_dummies(df_processed.Survived,drop_first=True).values 
df_processed.loc[:,["Sex"]] = pd.get_dummies(df_processed.Sex,drop_first=True).values
df_processed = pd.concat([df_processed,pd.get_dummies(df_processed.Pclass)],axis=1).drop(columns="Pclass") 
df_processed[df_processed.Fare == df_processed.Fare.max() ]
df_processed.drop(index=df_processed[df_processed.Fare>300].index,inplace=True)
#================Prepare data for ML training ===================================
df_train, df_test = train_test_split(df_processed,test_size = 0.3)

print('predict model for population')
#================Training ML model ===============================================
X_train = df_train.loc[:,['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 1, 2, 3]].values
y_train = df_train.Survived.values
X_test = df_test.loc[:,['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 1, 2, 3]].values
y_test = df_test.Survived.values
#==== Multi-Layer Perceptron (MLP) Model
nn_model = MLPClassifier(hidden_layer_sizes=(200,100),max_iter=5000)
nn_model.fit(X=X_train,y=y_train)
y_test_predicted_nn = nn_model.predict(X_test)
report_nn = classification_report(y_pred=y_test_predicted_nn,y_true=y_test)
print("Multi-Layer Perceptron")
print(report_nn)
#=======logistic Regression====================================
lr_model =LogisticRegression(random_state=0,max_iter=1000)
lr_model.fit(X=X_train,y=y_train)
y_test_predicted = lr_model.predict(X_test)
report_lr = classification_report(y_pred=y_test_predicted,y_true=y_test)
print("Linear_Regression")
print(report_lr)
#==================DecisionTree========================
dt_no_model = DecisionTreeClassifier().fit(X=X_train,y=y_train)
y_test_predicted_dt_no = dt_no_model.predict(X_test)
report_dt_no = classification_report(y_pred=y_test_predicted_dt_no,y_true=y_test)
print("DecisonTreeClassifier with no outliers")
print(report_dt_no)
df_tatanic_final.to_csv('file.csv',index=False)







