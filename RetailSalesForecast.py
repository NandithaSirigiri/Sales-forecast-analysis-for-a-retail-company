# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:59:01 2018

@author: Icheme
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:\\Users\\Icheme\\Desktop\\ML\\My-ML\\MultipleLinearRegression\\Retail Analysis_Dataset.csv'
retail = pd.read_csv(path)
retail.columns=retail.columns.str.strip()
print('retail',retail.columns)
r_1 = pd.DataFrame(retail)
print(r_1)


print(retail)
print(retail.shape)
print(retail.head())
print(retail.columns)

# removing the spaces in the headers in the columns
print(r_1.columns.tolist())
r_1.columns=r_1.columns.str.strip()
print(r_1.columns.tolist())

#Hence there are no missing values present in the data

#removing the dollar sign from the columns in the dataframe
r_1=r_1.replace({'\$':''},regex=True)
print('done')

# Coming to normalization, here there the scale in the dataset is not very large so no need for normalization to relate each attributes
#finding out the dataypes and correcitng it
print(r_1.dtypes)
#here the profit, shipping cost and Sales features are objects instead of float
r_1[['Sales','Shipping Cost','Profit']]=r_1[['Sales','Shipping Cost','Profit']].astype('float')
print(r_1.dtypes)
#r_1['Sales','Profit','Shipping Cost']=r_1['Sales','Profit','Shipping Cost'].remove('')
#Hence all the dtypes are assined correctly now.
#1.Order_ID - categorical
#2.Products -categorical
#3.Sales - numeric
#4.Quantity - categorical
#5. Discount-numeric
#6.Profit - numerci
#7.Shipping cost- numeric

correlation = r_1.corr()
print(correlation)

#feature engineering i.e., finding out what are the important varibales that affect the label


# In order to start understanding the (linear) relationship between an individual variable and the price. We can do this by using "regplot", which plots the scatterplot plus the fitted regression line for the data.

sns.regplot(x='Quantity',y='Sales',data=r_1)
plt.title('Qunatity vs Sales')
plt.show()


sns.regplot(x='Discount',y='Sales',data=r_1)
plt.title('Discount vs Sales')


plt.show()

sns.regplot(x='Profit',y='Sales',data=r_1)
plt.title('Profit vs Sales')
plt.show()

sns.regplot(x='Shipping Cost',y='Sales',data=r_1)
plt.title('Shipping Cost vs Sales')
plt.show()

sns.regplot(x='Order_ID',y='Sales',data=r_1)
plt.title('Order_ID vs Sales')
plt.ylim(0,250)
plt.xlim(10990,110050)
plt.show()

# here we can see that the Shipping cost, profit goes up the Sales also increses and are good precidtors of Sales  and the remaining Quantity and discoutn has a minimal effect in incresing the Sales
#We have only conseidered these variables ad they are the continous numarical variables and they also have the positive linear relationship with Sales and are g
print(r_1[['Quantity','Sales']].corr())

# Here the categorical features are Order Id and Products
#creating the dummy_varraibels for both

dummy_var_1=pd.get_dummies(r_1['Order_ID'])
dummy_var_2=pd.get_dummies(r_1['Products'])
print(dummy_var_1,dummy_var_2)

r_2=pd.DataFrame(r_1)
print(r_2)

r_2=pd.concat([r_2,dummy_var_1,dummy_var_2],axis=1)
print(r_2)



sns.boxplot(x='Products',y='Sales', data=r_1,width=0.8)
plt.xticks(rotation=90)
plt.ylim(0,280)
plt.show()


#here I have used the boxplot to understand on the sales for each product
#If necessary to comapre the sales in between the products we can check on them

# Descriptive statistical analysis

des_stats=r_1.describe(include='all')
print(des_stats)

print(r_1['Products'].unique())
print(r_1['Order_ID'].unique())
print(r_1['Products'].value_counts())
print(r_1['Order_ID'].value_counts())

Product_counts=r_1['Products'].value_counts().to_frame()
print(Product_counts.rename(columns={'Products':'Product_value_counts'}))
Product_counts.index.name="Products"
print(Product_counts.head())
#above we have found the number of product counts present
# Now let use see the sales of each Products groups

Rgroup=r_1[['Products','Sales']]

#finding the average price of each of the different categories of data
Rgroup=Rgroup.groupby(['Products'],as_index=False).mean()
print(Rgroup) 

#Finding out hte correlation and casuation between the variables 
# For correlation, we will use the pearson coefficinet,p_value
# here from the regression plots drand the four features given i.e, Shipping cost, quantity, disocunt and profit have showed some relationships
#finding out the correlations between the features and Label(Sales)
print(r_1.corr())
from scipy import stats
pearson_coef, p_value=stats.pearsonr(r_1['Shipping Cost'],r_1['Sales'])
print('the pearson coeff for Shiiping cost is',pearson_coef,'with p-value of',p_value)
# pearosn for shipping cost i positively linear,p_value<0.0001 strong evidence that correlation is significant


pearson_coef, p_valeu=stats.pearsonr(r_1['Quantity'],r_1['Sales'])
print('the pearson coeff for QUantity is',pearson_coef,'with p-value of',p_value)
#pearon coef for QUntity is almost equal to 0, indicting not it doesnot affect the Sales adnp-valeus alsoe signifies that

pearson_coef, p_value=stats.pearsonr(r_1['Discount'],r_1['Sales'])
print('the pearson coeff for Discount is',pearson_coef,'with p-value of',p_value)
#pearson coef Discount is negatively related,p-value> 0.1 shows there is no evidenc that the coreelation is significant

pearson_coef,p_valeu=stats.pearsonr(r_1['Profit'],r_1['Sales'])
print('the pearson coeff for Profit is',pearson_coef,'with p-value of',p_value)
#pearosn coeff is Profit is positvely significant but p-values>0.1 showing no evidence

pearson_coef,p_value=stats.pearsonr(r_1['Order_ID'],r_1['Sales'])
print('the pearson coeff for Order_ID is',pearson_coef,'with p-value of',p_value)
# pearson coeff for Order_ID isalmost not related to Sales,and p-value shows no evidence that hteres is significatn correlation

# hers first from the hypothesis testing based on the reg plot that we got initlayy
# Assumption made that Shipping cost is correlated to Sales with it true and null hypothesis is true, our hypothesis that Shipping cost affecting the Sales is true
#Simillarly with the Qunti, our hypothesis is true
#  For Discount, we assumed that they are slightly negatively related but no significant evidence that its true, our hypothesis for this is rejected
#For Profit. We have concluded that Sales incresed with increse in Profit but the p-value obtained explaine no evidence that the correlation is significant
#for Order_ID, no signnificant evidence

# Here we found that among the variables, only the Shiiping cost is affecting the SAles
# Let us develop the model for it

# will try to develop different models that predicts the price of the carusing the features
# the model will help us  the exact relationship between different variable
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
print(lm)
X=r_1[['Shipping Cost']]
Y=r_1[['Sales']]
lm.fit(X,Y)

Yhat=lm.predict(X)
print(Yhat[0:5])
print('intercept=',lm.intercept_)
print('coeff=',lm.coef_)
# here the estimated linaer model is Yhat=61.41738093+12.70341615*X
#Sales= 61.41738093+12.70341615*Shipping COst

lm_1=LinearRegression()
print(lm_1)
X1=r_1[['Quantity']]
Y1=r_1[['Sales']]
lm_1.fit(X1,Y1)
print('intercept for lm_1=',lm_1.intercept_)
print('coef for lm_1=',lm_1.coef_)
Yhat1=lm_1.predict(X1)
print(Yhat1[0:5])

#Sales= 138.54339623+4.55471698* Quantity

lm_2=LinearRegression()
print(lm_2)
X2=r_1[['Profit']]
Y2=r_1[["Sales"]]
lm_2.fit(X2,Y2)
print('intercept for lm_2 is', lm_2.intercept_,'the coef',lm_2.coef_)
Yhat2=lm_2.predict(X2)
print(Yhat2[0:5])

#Sales= 61.50352386 + 1.26832499* Profit


#Multibariate Linear regression
# Analysis one
lm_3 = LinearRegression()
z=r_1[['Shipping Cost','Quantity','Profit','Order_ID']]
lm_3.fit(z,r_1['Sales'])
print('intercept fir Multivaraite analysis', lm_3.intercept_,'coeff', lm_3.coef_)
Yhat_m=lm_3.predict(z)
print(Yhat_m[0:5])

# Analysis 2
lm_4 = LinearRegression()
z1=r_1[['Shipping Cost','Quantity','Profit']]
lm_4.fit(z1,r_1['Sales'])
print('intercept fir Multivaraite analysis', lm_4.intercept_,'coeff', lm_4.coef_)
Yhat_m1=lm_4.predict(z1)
print(Yhat_m1[0:5])
#Model Evaluation using Visualization¶
# Plotting a regression plot for the models

width=5
height=4
plt.figure(figsize=(width,height))
sns.regplot(x='Shipping Cost',y='Sales',data=r_1)
plt.ylim(0,)


# let use find the vairance of the data by using residual plot
#Residue is nothing but the fieerence between the (observed value=y) and predicted value(Yhat)
# In the regression plot, the residual is the distance between the data point and the regression line
#the residula plot shoes the residual on Y-axis and independent variable on x-axis
# let us look at the varibales in the resiudla plot

w=5
ht=6
plt.figure(figsize=(w,ht))
sns.residplot(r_1['Shipping Cost'],r_1['Sales'])
plt.show()

plt.figure(figsize=(w,ht))
sns.regplot(r_1['Quantity'],r_1['Sales'])
plt.show()

plt.figure(figsize=(w,ht))
sns.residplot(r_1['Profit'],r_1['Sales'])
plt.show()







#Here the residuals of the Shipping cost are notrandomly spread aorund x-axis lead us to belive may be linear regression is inappropritae
#Let us check for multiple linear regression
Y_hat=lm_3.predict(z)
print(Y_hat)

plt.figure(figsize=(w,ht))
ax3=sns.distplot(r_1['Sales'],hist=False, color='r', label='Actual value')
sns.distplot(Yhat,hist=False,color='b',label='Predicted',ax=ax3)

plt.figure(figsize=(w,ht))
ax1=sns.distplot(r_1['Sales'],hist=False,color='r',label='Actual Value')
sns.distplot(Yhat_m,hist=False,color='b',label='Predicted',ax=ax1)

plt.figure(figsize=(w,ht))
ax2=sns.distplot(r_1['Sales'],hist=False,color='r',label='Actual Value')
sns.distplot(Yhat_m1,hist=False,color='b',label='Predicted', ax=ax2)


##plt.figure(figsize=(w,ht))
#ax2=sns.distplot(r_1['Sales'],hist=False,color="r", label='Actual Value')
#sns.distplot(Yhat)


# Let us find the measures 
# FInding the R^2 scores
print('r2 score for LR with Shipping Cost', lm.score(X,Y))
print('r2 score for LR with Quantityt', lm_1.score(X1,Y1))
print('r2 score for LR with profit', lm_2.score(X2,Y2))

#r2 for multiple linear regression
print('r2 score for MLR -1', lm_3.score(z,r_1['Sales']))
print('r2 score for MLR -2', lm_4.score(z1,r_1['Sales']))


from sklearn.metrics import mean_squared_error
print(mean_squared_error(r_1['Sales'], Yhat))
print(mean_squared_error(r_1['Sales'], Yhat1))
print(mean_squared_error(r_1['Sales'], Yhat2))
print(mean_squared_error(r_1['Sales'], Yhat_m))
print(mean_squared_error(r_1['Sales'], Yhat_m1))


# Here we can check that the model Multiple Linear regression one has the higher R^2 value and the less mean squared error
# Hence the MLR-1 model is what we consider
# Last Step:Training and Testing the data

y_data=r_1['Sales']
x_data= z

from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test= train_test_split(x_data, y_data, test_size=0.25, random_state=1)
print('no of train sample', X_train.shape)
print('no of test sample',X_test.shape)

lre=LinearRegression()
lre.fit(X_train[['Shipping Cost','Quantity','Profit','Order_ID']],y_train)
print('R score for train ddata with z',lre.score(X_test[['Shipping Cost','Quantity','Profit','Order_ID']],y_test))

lre.fit(X_train[['Shipping Cost','Quantity','Profit']],y_train)
print('R score for train data with z1',lre.score(X_test[['Shipping Cost','Quantity','Profit']],y_test))

#Let us check for cross-validation

from sklearn.model_selection import cross_val_score
Rcross1=cross_val_score(lre,x_data[['Shipping Cost','Quantity','Profit','Order_ID']],y_data,cv=3)
Rcross2=cross_val_score(lre,r_1[['Shipping Cost','Quantity','Profit']],y_data,cv=3)
print('RXross1',Rcross1)
print('Rcross2',Rcross2)
print('Rcr0ss1 mean', Rcross1.mean(),'Rcross1 std', Rcross1.std())
print('Rcross2 mean', Rcross2.mean(), 'Rcross2 std',Rcross2.std())

from sklearn.model_selection import cross_val_predict

Yhat_m=cross_val_predict(lre,x_data[['Shipping Cost','Quantity','Profit','Order_ID']],y_data,cv=3)
print('Yhat_m',Yhat_m[0:5])
Yhat_m1=cross_val_predict(lre,r_1[['Shipping Cost','Quantity','Profit']],y_data,cv=3)
print('Yhat_m1',Yhat_m1[0:5])


# Checking for ovefittng , underfitting and model selection
lr=LinearRegression()
lr.fit(X_train[['Shipping Cost','Quantity','Profit','Order_ID']],y_train)

yhat_train1=lr.predict(x_data[['Shipping Cost','Quantity','Profit','Order_ID']])
print('yhat train values',yhat_train1[0:5]) 

yhat_test1=lr.predict(X_test[['Shipping Cost','Quantity','Profit','Order_ID']])
print('yhat teste values',yhat_test1[0:5])

lr.fit(X_train[['Shipping Cost','Quantity','Profit']],y_train)

yhat_train2=lr.predict(x_data[['Shipping Cost','Quantity','Profit']])
print('yhat train2 values',yhat_train2[0:5]) 

yhat_test2=lr.predict(X_test[['Shipping Cost','Quantity','Profit']])
print('yhat test2 values',yhat_test2[0:5])


def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title ):
    width = 6
    height = 5
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Sales')
    plt.ylabel('Proportion')

    plt.show()
    plt.close()
    
Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)
plt.show()

Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
DistributionPlot(y_train,yhat_train2,"Actual Values (Train)","Predicted Values (Train)",Title)
plt.show()


# Hence of all the modesl of linear regression and mutiplinear regression, we have foudn the the mulvariate regression
#with x_data[['Shipping Cost','Quantity','Profit', 'Order ID']
#with R^2 value and the less mean squared error

