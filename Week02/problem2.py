import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm


def OLS(data):
    x = data.x.values.reshape(-1,1)
    y = data.y.values.reshape(-1,1)
    regr = LinearRegression().fit(x, y)
    y_pred = regr.predict(x)

    
    sns.scatterplot(data, y = 'y',x = 'x')
    plt.plot(x,y_pred,color = 'orange')
    plt.title("Regression Fitting Plot")
    plt.show()

    error = y-y_pred
    sm.qqplot(error,line="45")
    plt.title("QQ Plot")
    plt.show()
    sns.histplot(error)
    plt.title("Error Histogram Plot")
    plt.show()
    

if __name__ == '__main__':
    
    data = pd.read_csv("problem2.csv")
    
    OLS(data)