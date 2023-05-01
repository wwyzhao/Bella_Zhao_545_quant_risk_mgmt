import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from RiskMgmt import PortfolioOptimization


def get_expected_return(stock_list, model_data, factor_data, factor_list, rf):
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

def portfolio_optimization(stock_list, model_data, exp_rtn_FFM, rf):
    """maximum sharpe to get the market portfolio weights 

    Args:
        stock_list (list): list of stock names
        model_data (df): FFM (return, FF3, Mom)
        exp_rtn_FFM (df): expected returns of each stock

    Returns:
        np.array: market portfolio weights
    """
    n_stock = len(stock_list)
    df_stocks = model_data[stock_list].astype(np.float64)
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

    return optimal_weight

def get_ex_post_attribution(optimal_weights, updated_returns, stock_list):
    """ex-post return and risk attribution

    Args:
        optimal_weights (np.array): results of portfolio_optimization
        updated_returns (df): stock returns df
        stock_list (list): stock name list
    """
    weightList = []
    lastW = np.array(optimal_weights)
    pReturn = []
    R = updated_returns.copy().reset_index()[stock_list].astype(np.float64)
    t = R.shape[0]
    for i in range(t):
        weightList.append(lastW)
        lastW = np.array(lastW * (1 + R.iloc[i,:])) 
        sumW = sum(lastW)
        lastW = lastW / sumW
        pReturn.append(sumW - 1)
    pReturn = np.array(pReturn)
    pstd= np.std(pReturn, ddof = 1)
    weights = pd.DataFrame(weightList, columns = stock_list) 
    totalReturn = np.exp(sum(np.log(pReturn + 1))) - 1
    # Carino K
    K = np.log(totalReturn + 1) / totalReturn
    carinoK = (np.log(pReturn + 1) / pReturn) / K
    # total return
    TR = [] 
    for col in R.items():
        tr = np.exp(sum(np.log(col[1] + 1))) - 1
        TR.append(tr)
    # return attribution
    ATR = [] 
    Y = R * weights 
    for col in Y.items():
        newCol = col[1] * carinoK
        ATR.append(sum(newCol))
    Attribution = pd.DataFrame({"TotalReturn": TR, "ReturnAttribution": ATR}, index = stock_list)
    # vol attribution
    X = np.array(sm.add_constant(pd.DataFrame({"pReturn": pReturn})))
    Y = np.array(Y)
    Beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1]
    cSD = Beta * np.std(pReturn, ddof = 1) 
    Attribution.insert(loc = 2, column = "VolAttribution", value = cSD)
    portfolio = pd.DataFrame({"TotalReturn": totalReturn, "ReturnAttribution": totalReturn, "VolAttribution": pstd}, index = ["Portfolio"])
    Attribution = pd.concat([Attribution, portfolio], axis = 0)
    print(Attribution) 

    return Attribution

def get_ex_post_attribution_ffm_alpha(optimal_weights, updated_returns, updated_FFM, factor_list_FFM, stock_list, Betas):
    lastW = np.array(optimal_weights)
    weightList = []
    pReturn = []
    residR = []
    factorWeights = []
    pReturn = []
    residR = []
    R = updated_returns.copy().reset_index()[stock_list].astype(np.float64)
    ffReturns = updated_FFM[factor_list_FFM] 
    t = R.shape[0]

    factorW = np.matrix(sum((Betas * lastW).T))
    for i in range(t):
        weightList.append(lastW)
        lastW = np.array(lastW * (1 + R.iloc[i,:])) 
        sumW = sum(lastW)
        lastW = lastW / sumW
        pReturn.append(sumW - 1)
        rR = (sumW - 1) - factorW * np.matrix(ffReturns.iloc[i,:]).T
        residR.append(rR[0,0])
    pReturn = np.array(pReturn)
    pstd = np.std(pReturn, ddof = 1)
    weights = pd.DataFrame(weightList, columns = stock_list) 
    totalReturn = np.exp(sum(np.log(pReturn + 1))) - 1
    updated_FFM.insert(loc = updated_FFM.shape[1], column = "Alpha", value = np.array(residR))

    # Carino K
    K = np.log(totalReturn + 1) / totalReturn
    carinoK = (np.log(pReturn + 1) / pReturn)/K

    factorWeights = factorW.repeat(t, 0).reshape(t, len(factor_list_FFM))
    r_carinoK = carinoK.repeat(4).reshape(factorWeights.shape)
    Attrib = ffReturns * factorWeights * r_carinoK
    residCol = residR * carinoK
    Attrib.insert(loc = Attrib.shape[1], column = "Alpha", value = residCol)

    new_factor_list = factor_list_FFM + ["Alpha"]
    # total return
    TR = []
    for col in updated_FFM[new_factor_list].items():
        tr = np.exp(sum(np.log(col[1] + 1))) - 1
        TR.append(tr)
    # print(TR)
    # return attribution
    ATR = []
    for col in Attrib.items():
        ATR.append(sum(col[1]))
    # print(ATR)
    Attribution = pd.DataFrame({"TotalReturn": TR, "ReturnAttribution": ATR}, index = new_factor_list)
    # vol attribution
    Y = ffReturns * factorWeights
    Y.insert(loc = Y.shape[1], column = "Alpha", value = np.array(residR))
    X =  np.array(sm.add_constant(pd.DataFrame({"pReturn": pReturn})))
    Y = np.array(Y)
    Beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1]
    cSD = Beta * np.std(pReturn, ddof = 1)
    Attribution.insert(loc = 2, column = "VolAttribution", value = cSD)
    portfolio = pd.DataFrame({"TotalReturn": totalReturn, "ReturnAttribution": totalReturn, "VolAttribution": pstd}, index = ["Portfolio"])
    Attribution = pd.concat([Attribution, portfolio], axis = 0)
    print(Attribution) 

    return Attribution
