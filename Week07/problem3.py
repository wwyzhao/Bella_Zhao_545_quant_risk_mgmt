import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from RiskMgmt import VaR


stock_list = ["AAPL", "META", "UNH", "MA",
                 "MSFT", "NVDA", "HD", "PFE",
                 "AMZN", "BRK-B", "PG", "XOM",
                 "TSLA", "JPM", "V", "DIS",
                 "GOOGL", "JNJ", "BAC", "CSCO"]

factor_list_FF3 = ["Mkt-RF", "SMB", "HML"]
factor_list_FFM = ["Mkt-RF", "SMB", "HML", "Mom"]
rf = 0.0425

def get_expected_return(stock_list, model_data, factor_data, factor_list):
    dict_param = {}
    # OLS for each stock to get parameters of factor returns
    for r_name in stock_list:
        # OLS:
        #   y: r - rf 
        #   constant: alpha 
        #   x1~xk: factor returns
        X = model_data[factor_list]
        X = sm.add_constant(X).astype(np.float64)
        Y = (model_data[r_name] - model_data['RF']).astype(np.float64)
        regression = sm.OLS(Y, X)        
        ols_model = regression.fit()
        # print(ols_model.params)
        param = {}
        param["alpha"] = ols_model.params["const"]
        for f in factor_list:
            param[f] = ols_model.params[f] 
        dict_param[r_name] = param
    # print(dict_param)
    
    # expected return for each factor, i.e. risk premium
    factor_return = {}
    for f in factor_list:
        factor_return[f] = factor_data[f].mean()
    # expected risk free rate
    # dailyRF = model_data["RF"].mean()

    # calculate expected return for each stock using factor returns
    dict_r = {}
    for r_name in stock_list:
        r_daily = 0
        for f in factor_list:
            r_daily += dict_param[r_name][f] * factor_return[f]
        # r_daily += dict_param[r_name]["alpha"] # assume alpha to be 0 in the long term
        #Discrete Returns, convert to Log Returns and scale to 1 year
        r_annual = np.log(r_daily + 1) * 255 + rf
        dict_r[r_name] = r_annual
    df_r = pd.DataFrame(dict_r, index = [0]).T
    df_r.columns = ["return"]
    return df_r

# clean the data and calculate daily returns
ff3 = pd.read_csv("F-F_Research_Data_Factors_daily.csv")
mom = pd.read_csv("F-F_Momentum_Factor_daily.csv")
prices = pd.read_csv("DailyPrices.csv", index_col="Date")
# eliminate white spaces
col_names = mom.columns.tolist()
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
mom.columns=col_names 
# calculate daily returns
returns = VaR.return_calculate(prices)
returns = returns[1:].reset_index() # add date to column
returns['Date'] = pd.to_datetime(returns['Date']).dt.strftime('%m/%d/%Y')
# convert the date format
ff3["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), ff3["Date"]))
returns["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), returns["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))
# get past 10 years history data
startdate_str = "20130131"
enddate_str = "20230201"
ff3 = ff3[ff3["Date"] > datetime.datetime.strptime(startdate_str, "%Y%m%d")]
ff3 = ff3[ff3["Date"] < datetime.datetime.strptime(enddate_str, "%Y%m%d")]
returns = returns[returns["Date"] > datetime.datetime.strptime(startdate_str, "%Y%m%d")]
returns = returns[returns["Date"] < datetime.datetime.strptime(enddate_str, "%Y%m%d")]
mom = mom[mom["Date"] > datetime.datetime.strptime(startdate_str, "%Y%m%d")]
mom = mom[mom["Date"] < datetime.datetime.strptime(enddate_str, "%Y%m%d")]
# divided by 100 to get like units
ff3[factor_list_FF3] = ff3[factor_list_FF3] / 100 
mom["Mom"] = mom["Mom"] / 100
# align the dates, get the dataframe for the regressions
FF3 = pd.merge(returns, ff3, on = "Date", how = "left")
FFM = pd.merge(FF3, mom, on = "Date", how = "left")
factor_data_FF3 = ff3.copy()
factor_data_FFM = pd.merge(factor_data_FF3, mom).copy()

# find the expected annual return based on the past 10 years of factor returns
# exp_rtn_FF3 = get_expected_return(stock_list, model_data = FF3, factor_data = factor_data_FF3, factor_list = factor_list_FF3)
exp_rtn_FFM = get_expected_return(stock_list, model_data = FFM, factor_data = factor_data_FFM, factor_list = factor_list_FFM)
# summary = exp_rtn_FF3.rename(columns={'return': 'estimated annual return by FF3 model (%)'}).join(exp_rtn_FFM.rename(columns={'return': 'estimated annual return by FF-M model (%)'}))
# print(summary * 100)


# portfolio construction
n_stock = len(stock_list)
df_stocks = FFM[stock_list].astype(np.float64)
# annual covariance matrix
sigma = np.matrix((np.log(df_stocks + 1)).cov() * 255)
exp_rtn = np.matrix((exp_rtn_FFM["return"]).values)

def cal_sharpe(w, rtn, cov, rf):
    w = np.matrix(w)
    r_p = w * rtn.T
    s_p = np.sqrt(w * cov * w.T)
    sharpe = (r_p[0,0] - rf) / s_p[0,0]
    return -sharpe
x0 = np.array([1 / n_stock] * n_stock) # initialized weights
args = (exp_rtn, sigma, rf)
bound = [(0.0, 1) for _ in stock_list]
cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
result = minimize(cal_sharpe, x0 = x0, args = args, bounds = bound, constraints = cons)

optimal_weight = result.x
market_port = pd.DataFrame({"Stock": stock_list,"weights(%)": [round(x, 4) for x in (optimal_weight * 100)]})
market_port["Expected return"] = exp_rtn_FFM["return"].values
print("Market portfolio weights and Expected annual return by FF-M model")
print(market_port)
print("Expected Return = ", market_port["Expected return"].T @ optimal_weight)
print("Expected Vol = ", np.array(np.sqrt(optimal_weight.T.reshape((n_stock,1)).T * sigma * optimal_weight.T.reshape((n_stock, 1)))).squeeze())
print("Expected Sharpe = ", -cal_sharpe(np.array(optimal_weight), np.matrix(market_port["Expected return"].values), sigma, rf))