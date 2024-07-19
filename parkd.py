# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


data=pd.read_csv('D:\SKILL DEVLP\HACKATHONS\GNIT HACK\Parkinsson disease.csv')

data.head()

data.tail()

# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
data.info()

data.describe()

data.shape

data.isnull().sum()

sns.heatmap(data.corr())
sns.set(font_scale=0.25)
plt.show()

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 60

features = ['status']

for i, feat in enumerate(features):
    plt.subplot(4, 2, i + 1)
    sns.distplot(data[feat], color='greenyellow')
    if i < 3:
        plt.title(f'Ratio of {feat}', fontsize=12)
    else:
        plt.title(f'Distribution of {feat}', fontsize=12)
    plt.tight_layout()
    plt.grid()

x=['name']
y=['status']
plt.scatter(x, y)
plt.title('Scatter Plot Example')
plt.xlabel('name')
plt.ylabel('status')
# Show the plot
plt.show()

data['status'].value_counts()

data.groupby('status').mean()

x=data.drop(columns=['name','status'],axis=1)
y=data['status']
x

y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

ss=StandardScaler()
ss.fit(x_train)

x_train=ss.transform(x_train)
x_test=ss.transform(x_test)
print(x_train)

print(x_test)

model=svm.SVC(kernel='linear')

model.fit(x_train,y_train)

x_train_pred=model.predict(x_train)
train_data_acc = accuracy_score(y_train,x_train_pred)

print("acc of training model :",train_data_acc)

x_test_pred=model.predict(x_test)
test_data_acc = accuracy_score(y_test,x_test_pred)

print("acc of testing model :",test_data_acc)

input_data=(197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569)
input_data_np=np.asarray(input_data)
input_data_re=input_data_np.reshape(1,-1)
S_data=ss.transform(input_data_re)
pred=model.predict(S_data)
print(pred)
if(pred[0]==0):
  print("Negative,No parkinson's found")
else:
    print("Positive, parkinson's found")
    
import pickle
pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569]]))
'''
input_data=(197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569)
input_data_np=np.asarray(input_data)
input_data_re=input_data_np.reshape(1,-1)
S_data=ss.transform(input_data_re)
pred=model.predict(S_data)
print(pred)
if(pred[0]==0):
  print("Negative,No parkinson's found")
else:
    print("Positive, parkinson's found")
'''
# Commented out IPython magic to ensure Python compatibility.
# %%shell
# jupyter nbconvert --to html /content/parkd.ipynb