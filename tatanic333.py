"""
Created on Fri Jan 10 21:01:33 2020

@author: Ashiat Adeogun
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#%matplotlib inline


df_tatanic =pd.read_csv('titanic.csv',sep=',',index_col='PassengerId')

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
             age_group ='Adult'
        else:
             age_group='Elderly'
        Age_group.append(age_group)
    x['Age_group'] =  Age_group
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


#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    tr = split_name(x)
    tr=pd.DataFrame(tr)
    title=tr['title']
    sex =tr['Sex']
    for i in range(1,len(title)+1):
        if title[i] in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            title[i] = 'Mr'
        elif title[i]  in ['Countess', 'Mme']:
            title[i] = 'Mrs'
        elif title[i]  in ['Mlle', 'Ms']:
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
    df_tatanic['Sex']=df_tatanic['Sex'].str.title()#modify Sex column male and female by changing the first letters into capital letter
    df_tatanic['Survived']=df_tatanic['Survived'].replace({1:'Yes',0:'No'})
    df_tatanic['Age']=df_tatanic['Age'].fillna(29.7)#replace the NA values by the mean value for the age column
    return x

def final_process(x):
    df_1=age_group(x)
    df_2=embark(df_1)
    df_3=pclass(df_2)
    df_4=split_name(df_3)
    df_5=replace_titles(df_4)
    df_6=edit_colum(df_5)
    return df_6

df_tatanic_final = final_process(df_tatanic)


#df_tatanic_final.isna().sum()#to check for na value
#df_tatanic_final['Age'].isna().value_counts()#there  are 177 NAvalues
#df_tatanic_final['Age'].describe()#29.7 of mean

#function to  plot visualize categorical variable
def plot_bar(column,df_tatanic_final):
    # temp df 
    temp_1 = pd.DataFrame()#empty df
    # count categorical values
    temp_1['Yes_Survied'] = df_tatanic_final[df_tatanic_final['Survived'] == 'Yes'][column].value_counts()
    temp_1['No_Survived'] = df_tatanic_final[df_tatanic_final['Survived'] == 'No'][column].value_counts()
    temp_1.plot(kind ='bar')
    plt.xlabel(f'{column}')
    plt.ylabel('Number of survived')
    plt.title('Distribution of {} and Survived'.format(column))
    plt.show()
    return temp_1

plot_bar('Sex',df_tatanic_final)
plot_bar('Embarked',df_tatanic_final)
plot_bar('Pclass',df_tatanic_final)
plot_bar('Age_group',df_tatanic_final)


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

df_tatanic_final.to_csv('file.csv',index=False)










