import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from RiskMgmt import PortfolioOptimization, VaR


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
# divided by 100 to get like units
ff3[factor_list_FF3] = ff3[factor_list_FF3] / 100 
mom["Mom"] = mom["Mom"] / 100
# get past 10 years history data
startdate_str = "20120113"
enddate_str = "20220115"
ff3_ = ff3[ff3["Date"] > datetime.datetime.strptime(startdate_str, "%Y%m%d")]
ff3_ = ff3_[ff3_["Date"] < datetime.datetime.strptime(enddate_str, "%Y%m%d")]
returns_ = returns[returns["Date"] > datetime.datetime.strptime(startdate_str, "%Y%m%d")]
returns_ = returns_[returns_["Date"] < datetime.datetime.strptime(enddate_str, "%Y%m%d")]
mom_ = mom[mom["Date"] > datetime.datetime.strptime(startdate_str, "%Y%m%d")]
mom_ = mom_[mom_["Date"] < datetime.datetime.strptime(enddate_str, "%Y%m%d")]
# align the dates, get the dataframe for the regressions
FF3 = pd.merge(returns_, ff3_, on = "Date", how = "left")
FFM = pd.merge(FF3, mom_, on = "Date", how = "left")
factor_data_FF3 = ff3_.copy()
factor_data_FFM = pd.merge(factor_data_FF3, mom_).copy()

exp_rtn_FFM = PortfolioOptimization.get_expected_return(stock_list, FFM, factor_data_FFM, factor_list_FFM, rf)
optimal_weights = PortfolioOptimization.portfolio_optimization(stock_list, FFM, exp_rtn_FFM, rf)

# ex-post return and risk attribution for each stock
updated_prices = pd.read_csv("updated_prices.csv", index_col="Date")
updated_ff3 = pd.read_csv("updated_F-F_Research_Data_Factors_daily.csv")
updated_mom = pd.read_csv("updated_F-F_Momentum_Factor_daily.csv")
# calculate daily returns
updated_returns = VaR.return_calculate(updated_prices)
updated_returns = updated_returns[1:].reset_index() # add date to column
updated_returns['Date'] = pd.to_datetime(updated_returns['Date']).dt.strftime('%m/%d/%Y')
# eliminate white spaces in column names
col_names = updated_mom.columns.tolist()
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
updated_mom.columns = col_names 
# convert the date format
updated_ff3["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), updated_ff3["Date"]))
updated_returns["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), updated_returns["Date"]))
updated_mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), updated_mom["Date"]))
# divided by 100 to get like units
updated_ff3[factor_list_FF3] = updated_ff3[factor_list_FF3] / 100 
updated_mom["Mom"] = updated_mom["Mom"] / 100
# align the dates, get the dataframe for the regressions
ffm = pd.merge(ff3, mom, on = "Date", how = "left") # ffm here must be the data not been filtered by the past 10-year date
updated_factor_data_FFM = pd.merge(updated_ff3, updated_mom, on = "Date", how = "left").dropna(axis = 0)
all_FFM = pd.concat([ffm, updated_factor_data_FFM], axis = 0).reset_index(drop = True)
updated_FFM = pd.merge(updated_returns, all_FFM, on = "Date", how = "left").reset_index(drop = True)

# attribute realized risk and return to the Fama French 3+Momentum model, Report the residual total as Portfolio Alpha
# calculate portfolio betas first, need to transpose this "Betas"
X = sm.add_constant(FFM[factor_list_FFM]).astype(np.float64)
Y = FFM[stock_list].astype(np.float64)
Betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1:]
return_risk_attribution = PortfolioOptimization.get_ex_post_attribution_ffm_alpha(optimal_weights, updated_returns, updated_FFM, factor_list_FFM, stock_list, Betas)
