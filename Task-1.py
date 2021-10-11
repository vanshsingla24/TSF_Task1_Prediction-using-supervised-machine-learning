#!/usr/bin/env python
# coding: utf-8

# # Vansh Singla - Task 1:Prediction using supervised machine learning
# TASK OBJECTIVE: PREDICT THE PERCENTAGE OF STUDENTS BASED ON NO. OF HOURS OF STUDY AND ALSO PREDICT THE SCORE IF A STUDENT STUDIES FOR 9.25 HRS/DAY
# 
# 

# In[4]:


import numpy as np
import pandas as pd
import scipy 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
sns.set()


# In[5]:


# LOAD AND READ THE DATA
source = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data = pd.read_csv(source)
data.head(6)


# In[6]:


# GET A BRIEF DESCRIPTION OF THE DATA
data.info()


# In[7]:


data.describe()


# In[8]:


# VISUALIZE THE DATASET BY INTRODUCING DEPENDENT AND INDEPENDENT VARIABLE 
y = data['Scores']
x1 = data['Hours']
plt.scatter(x1,y)
plt.xlabel('Hours',fontsize=20)
plt.ylabel('Scores',fontsize=20)
plt.title('Hours vs Scores',fontsize=30)
plt.show()


# In[9]:



# SEPERATE THE DEPENDENT AND INDEPENDENT VARIABLE FOR REGRESSION
x1 = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[10]:



# SPLIT TRAINING AND TESTING DATA TO CHECK FOR OVERFITTING
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y,train_size = 0.75,random_state=42)


# In[11]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[12]:


# PREFORMING SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)


# In[13]:



# PLOT THE REGRESSION LINE
slope = reg.coef_
intercept = reg.intercept_

regline = slope*x1 + intercept

plt.scatter(x1,y)
plt.plot(x1,regline)
plt.show()


# In[14]:



# PERFORM THE REGRESSION IN THE TEST DATA AND COMPARE ACCURACY WITH TRAIN ACCURACY
a = reg.score(x_train,y_train)*100
b = reg.score(x_test,y_test)*100

print("Training Accuracy = ",a)
print("Test Accuract = ",b)


# In[15]:



# COMPARE THE ACTUAL AND PREDICTED VALUES FOR THE TEST DATASET 

y_predicted = reg.predict(x_test)
df = pd.DataFrame({"y_Actual" : y_test, "y_Predicted" : y_predicted})
df


# In[16]:



# EVALUATE THE MODEL

from sklearn import metrics

print("Mean Squared Error : ", metrics.mean_squared_error(y_test,y_predicted))
print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test,y_predicted))
print("The R^2 value is : ",reg.score(x_test,y_test))


# # WHAT WILL BE THE SCORE IF A STUDENT STUDIES FOR 9.25 HRS/DAY?

# In[18]:


pred = reg.predict([[9.25]])
print("\033[1m" + "Conclusion :")
print("For 9.25 hours of study per day,model predicts that the student will score",pred[0])


# In[ ]:




