import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from RiskMgmt import PortfolioOptimization, VaR, Simulation, ES


stock_list = ["AAPL", "MSFT", "BRK-B", "CSCO", "JNJ"]

factor_list_FF3 = ["Mkt-RF", "SMB", "HML"]
factor_list_FFM = ["Mkt-RF", "SMB", "HML", "Mom"]
rf = 0.0025

# clean the data and calculate daily returns
ff3 = pd.read_csv("F-F_Research_Data_Factors_daily.csv")
mom = pd.read_csv("F-F_Momentum_Factor_daily.csv")
returns = pd.read_csv("DailyReturn.csv")
# eliminate white spaces
col_names = mom.columns.tolist()
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
mom.columns=col_names 
# convert the date format
ff3["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), ff3["Date"]))
returns["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), returns["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))
# get past 10 years history data
startdate_str = "20120113"
enddate_str = "20220115"
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

exp_rtn_FFM = PortfolioOptimization.get_expected_return(stock_list, FFM, factor_data_FFM, factor_list_FFM, rf)
optimal_weights = PortfolioOptimization.portfolio_optimization(stock_list, FFM, exp_rtn_FFM, rf)

df_stocks = FFM[stock_list].astype(np.float64)
covar = np.matrix((np.log(df_stocks + 1)).cov() * 255)
# Function for Portfolio Volatility
def pVol(w):
    pvol = (w.T * covar * w)[0, 0]
    return np.sqrt(pvol)
# Function for Component Standard Deviation
def pCSD(w):
    pvol = pVol(w)
    csd = np.multiply(np.multiply(w, np.dot(covar, w)), 1/pvol)
    return csd    
def riskBudget(w):
    pSig = pVol(w)
    CSD = pCSD(w)
    rb = np.multiply(CSD, 1/pSig)
    return rb
riskBudgetOpt = pd.DataFrame(riskBudget(np.matrix(optimal_weights).T), index = stock_list)

# Sum Square Error of cSD
def sseCSD(w):
    w = np.matrix(w).T
    n = w.shape[0]
    csd = pCSD(w)
    mCSD = np.sum(csd) / n
    dCSD = csd - mCSD
    se = np.multiply(dCSD,dCSD)
    return np.sum(se) * 100000
nStocks = len(stock_list)
x0 = np.array(nStocks * [1 / nStocks])
bound = [(0.0, 1) for _ in stock_list]
cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
result = minimize(sseCSD, x0 = x0, bounds = bound, constraints = cons)
riskParityW = result.x
riskBudget(np.matrix(riskParityW).T)

# RP on Simulated ES
# Fit a Gaussian Copula T distribution
Y = FFM[stock_list]
sim_Y = Simulation.copula_t(Y)
print(sim_Y)


# internal ES function
def pES(w):
    r = np.dot(sim_Y, w.T)
    r = np.array(r)
    var_, es_ = ES.get_VaR_ES(r, 0.05)
    return es_
# Function for the component ES
def CES(w):
    x = w
    n = w.shape[0]
    es = pES(w)
    ces = np.zeros(n)
    e = 1e-6
    for i in range(n):
        old = x[i]
        x[i] = x[i] + e
        ces[i] = old * (pES(x) - es) / e
        x[i] = old
    return ces
# SSE of the Component ES
def sseCES(w):
    ces = CES(w)
    ces_m = ces - np.mean(ces)
    return np.dot(ces_m.T, ces_m) * 10000
nStocks = len(stock_list)
x0 = np.array(nStocks*[1 / nStocks])
bound = [(0.0, 1) for _ in stock_list]
cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
result = minimize(sseCES, x0 = x0, bounds = bound, constraints = cons)

ESriskParityW = result.x

summ = pd.DataFrame({"ES RP Portfolio": ESriskParityW, "Vol RP Portfolio": riskParityW}, index = stock_list)
print(summ)