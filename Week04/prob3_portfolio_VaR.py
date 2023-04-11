import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.tsa.arima.model as sm
import matplotlib.pyplot as plt
import seaborn as sns
from problem2 import return_calculate

# using the two csv files to get the daily total value of each portfolio, preparing for calculating returns
def parse_data(p, df):
    tot_value = pd.DataFrame(index=df.index, columns=["Total Value"])
    m = df.shape[0]

    # for data on each date
    for i in range(m):
        current = pd.DataFrame({"Stock":df.columns, "price":df.iloc[i,:]}) 
        #get the prices and holdings of each stock, calculate daily total price of all the holdings in a portfolio
        temp = pd.merge(current, p, on="Stock", how="inner")    
        temp_sum = sum(temp["price"] * temp["Holding"])
        tot_value.iloc[i][0] = temp_sum
    return tot_value

# Calculate VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)
def get_VaR_exp_w(rt, alpha, lambd):
    n = len(rt)
    weights = np.zeros(n)
    for i in range(n): # calculate weights array reversely
        weights[i] = (1 - lambd) * lambd ** (i - 1)
    weights = weights / sum(weights)
    weights = np.flip(weights) # reverse the weights array from n-1 to 0
    sigma_2 = sum(weights * rt * rt) # np.mean(rt) = 0
    # sigma_2 = weights.reshape(1, n) @ (rt * rt).reshape(n, 1)

    mu = 0
    sigma = np.sqrt(sigma_2)
    VaR = -st.norm.ppf(alpha, loc=mu, scale=sigma) # mu = 0, symmetric

    plt.cla()
    sns.histplot(rt, kde=True, color="blue")
    np.random.seed(1)
    x = np.random.normal(loc=0, scale=sigma, size=len(rt))
    sns.histplot(x, kde=True, color="orange")
    plt.title("VaR_exp_w")
    plt.savefig("plots/problem3_VaR_exp_w.png")

    return VaR

# Calculate VaR using a fitted AR(1) model
def get_VaR_AR_1(rt, alpha):
    # fit AR(1) model to return
    model = sm.ARIMA(rt, order=(1, 0, 0)).fit()

    # autoregressive coefficient and error variance
    ar_coef = model.arparams[0]
    err_var = model.mse

    # simulate returns using AR(1) model
    n = 1000
    np.random.seed(1)
    r = np.random.normal(loc=0, scale=1, size=n-1)
    sim_rt = np.zeros(n)
    sim_rt[0] = rt[-1]
    for i in range(1, n):
        sim_rt[i] = ar_coef * sim_rt[i - 1] + np.sqrt(err_var) * r[i - 1]
    VaR = -np.percentile(sim_rt, alpha * 100)

    plt.cla()
    sns.histplot(rt, kde=True, color="blue")
    np.random.seed(1)
    sns.histplot(sim_rt, kde=True, color="orange")
    plt.title("VaR_AR_1")
    plt.savefig("plots/problem3_VaR_AR_1.png")

    return VaR


if __name__ == '__main__':

    df = pd.read_csv("DailyPrices.csv", index_col="Date")
    port = pd.read_csv("portfolio.csv").groupby("Portfolio")

    # total value of 3 portfolios on each day
    tot_port_value = pd.DataFrame(index=df.index, columns=["Total Port Value"])
    tot_port_value["Total Port Value"] = 0
    price_latest_value = [] # latest price of each portfolio

    # Arithmetric Return and Exponentially Weighted variance (λ = 0. 94) VaR
    # for each portfolio, get the daily total value, calculate Arithmetric Return
    rt_port_list = [] # return list of 3 portfolios
    for p in port:
        tot_value = parse_data(p[1], df)
        tot_port_value["Total Port Value"] += tot_value["Total Value"] # get the total value of 3 portfolios on each day
        price_latest_value.append(tot_value.iloc[-1]["Total Value"])
        
        # return for each of the 3 protfolios, mean = 0
        rt_ARITH = return_calculate(tot_value, "ARITH_RT")
        rt_ARITH = rt_ARITH.loc[:, "Total Value"]
        rt_ARITH = np.array(rt_ARITH[1:]).astype(np.float64)
        rt_ARITH = rt_ARITH - np.mean(rt_ARITH)
        rt_port_list.append(rt_ARITH)

    # return for total 3 portfolios, mean = 0
    rt_tot_ARITH = return_calculate(tot_port_value, "ARITH_RT")
    rt_tot_ARITH = rt_tot_ARITH.loc[:, "Total Port Value"]
    rt_tot_ARITH = np.array(rt_tot_ARITH[1:]).astype(np.float64)
    rt_tot_ARITH = rt_tot_ARITH - np.mean(rt_tot_ARITH)

    alpha = 0.05
    lambd = 0.94
    VaR_exp_w = get_VaR_exp_w(rt_port_list[0], alpha, lambd)
    print("Portfolio A:")  
    print(f"VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94): {VaR_exp_w}, ${price_latest_value[0] * VaR_exp_w}")
    VaR_exp_w = get_VaR_exp_w(rt_port_list[1], alpha, lambd)
    print("Portfolio B:")  
    print(f"VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94): {VaR_exp_w}, ${price_latest_value[1] * VaR_exp_w}")
    VaR_exp_w = get_VaR_exp_w(rt_port_list[2], alpha, lambd)
    print("Portfolio C:")  
    print(f"VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94): {VaR_exp_w}, ${price_latest_value[2] * VaR_exp_w}")
    VaR_exp_w = get_VaR_exp_w(rt_tot_ARITH, alpha, lambd)
    print("VaR of total holdings")  
    print(f"VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94): {VaR_exp_w}, ${sum(price_latest_value) * VaR_exp_w}")
    

    # Geometric Brownian Motions and fitted AR(1) model VaR
    # for each portfolio, get the daily total value, calculate Arithmetric Return
    rt_port_list1 = [] # return list of 3 portfolios
    for p in port:
        tot_value = parse_data(p[1], df)
        tot_port_value["Total Port Value"] += tot_value["Total Value"] # get the total value of 3 portfolios on each day
        
        # return for each of the 3 protfolios, mean = 0
        rt_GBM = return_calculate(tot_value, "GBM")
        rt_GBM = rt_GBM.loc[:, "Total Value"]
        rt_GBM = np.array(rt_ARITH[1:]).astype(np.float64)
        rt_GBM = rt_GBM - np.mean(rt_GBM)
        rt_port_list1.append(rt_GBM)

    # return for total 3 portfolios, mean = 0
    rt_tot_GBM = return_calculate(tot_port_value, "GBM")
    rt_tot_GBM = rt_tot_GBM.loc[:, "Total Port Value"]
    rt_tot_GBM = np.array(rt_tot_GBM[1:]).astype(np.float64)
    rt_tot_GBM = rt_tot_GBM - np.mean(rt_tot_GBM)

    alpha = 0.05
    
    VaR_AR_1 = get_VaR_AR_1(rt_port_list[0], alpha)
    print("Portfolio A:")  
    print(f"VaR using a normal distribution with fitted AR(1) model: {VaR_AR_1}, ${price_latest_value[0] * VaR_AR_1}")
    VaR_AR_1 = get_VaR_AR_1(rt_port_list[1], alpha)
    print("Portfolio B:")  
    print(f"VaR using a normal distribution with fitted AR(1) model: {VaR_AR_1}, ${price_latest_value[1] * VaR_AR_1}")
    VaR_AR_1 = get_VaR_AR_1(rt_port_list[2], alpha)
    print("Portfolio C:")  
    print(f"VaR using a normal distribution with fitted AR(1) model: {VaR_AR_1}, ${price_latest_value[2] * VaR_AR_1}")
    VaR_AR_1 = get_VaR_AR_1(rt_tot_ARITH, alpha)
    print("VaR of total holdings")  
    print(f"VaR using a normal distribution with fitted AR(1) model: {VaR_AR_1}, ${sum(price_latest_value) * VaR_AR_1}")
