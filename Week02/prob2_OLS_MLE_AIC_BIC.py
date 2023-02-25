import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import minimize

def R_sqaure(y, y_pred):
    mu = y_pred.mean()
    sst = (y - mu).T @ (y - mu)
    sse = (y - y_pred).T @ (y - y_pred)
    R2 = 1 - sse / sst
    return R2

def OLS(data):
    x = data.x.values.reshape(-1, 1)
    y = data.y.values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    print(f"OLS fitting: y_pred = {reg.coef_} * x + {reg.intercept_}")
    y_pred = reg.predict(x)
    error = y - y_pred

    sns.scatterplot(data, y='y',x='x')
    plt.plot(x, y_pred, color='orange')
    plt.title("OLS Regression Fitting")
    plt.show()

    sns.histplot(error, kde=True)
    plt.title("OLS Error Histogram")
    plt.show()

    R_square = R_sqaure(y, y_pred)
    print(f"OLS fitting: R_Squared = {R_square}")

def MLE_Norm(parameters, x, y):
    beta, const, std = parameters
    y_pred = beta * x + const
    LL = np.sum(stats.norm.logpdf(y - y_pred, loc=0, scale=std))
    return -LL
def MLE_T(parameters, x, y):
    beta, const, std, degree = parameters
    y_pred = beta * x + const
    LL = np.sum(stats.t.logpdf(y - y_pred, loc=0, scale=std, df=degree))
    return -LL

def AIC_BIC_Norm(y, y_pred, std):
    k = 3
    LL = np.sum(stats.norm.logpdf(y - y_pred, loc=0, scale=std))
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(len(y)) - 2 * LL
    return AIC, BIC

def AIC_BIC_T(y, y_pred, std, degree):
    k = 4
    LL = np.sum(stats.t.logpdf(y - y_pred, loc=0, scale=std, df=degree))
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(len(y)) - 2 * LL
    return AIC, BIC

def MLE(data):
    x = data.x.values.reshape(-1, 1)
    y = data.y.values.reshape(-1, 1)


    #MLE fitting assumed normality
    mle_norm = minimize(MLE_Norm, np.array([2,2,2]), args = (x, y))
    beta_norm, const_norm, std_norm = mle_norm.x
    y_pred_norm = beta_norm * x + const_norm
    print(f"MLE fitting assumed normality: y_pred = {beta_norm} * x + {const_norm}")
    error_norm = y - y_pred_norm

    # MLE fitting assumed T-distribution
    mle_t = minimize(MLE_T, np.array([2,2,2,2]), args=(x, y))
    beta_t, const_t, std_t, degree_t = mle_t.x
    y_pred_t = beta_t * x + const_t
    print(f"MLE fitting assumed T-distribution: y_pred = {beta_t} * x + {const_t}")
    error_t = y - y_pred_t

    sns.scatterplot(data, y='y', x='x')
    plt.plot(x, y_pred_norm, color='orange')
    plt.plot(x, y_pred_t, color='green')
    plt.title("MLE fitting")
    plt.show()

    sns.histplot(error_norm, kde=True)
    plt.title("MLE Error_Norm Histogram")
    plt.show()
    sns.histplot(error_t, kde=True)
    plt.title("MLE Error_T Histogram")
    plt.show()


    # AIC BIC R-Squared
    AIC_Norm, BIC_Norm = AIC_BIC_Norm(y, y_pred_norm, std_norm)
    R_square_Norm = R_sqaure(y, y_pred_norm)
    AIC_T, BIC_T = AIC_BIC_T(y, y_pred_t, std_t, degree_t)
    R_square_T = R_sqaure(y, y_pred_t)
    print(f"MLE fitting assumed normality: AIC = {AIC_Norm}, BIC = {BIC_Norm}, R_Squared = {R_square_Norm}")
    print(f"MLE fitting assumed T-distribution: AIC = {AIC_T}, BIC = {BIC_T}, R_Squared = {R_square_T}")


if __name__ == '__main__':
    
    data = pd.read_csv("problem2.csv")
    
    OLS(data)

    MLE(data)