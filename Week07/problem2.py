import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as sm
import datetime
import scipy.stats as st
from RiskMgmt import VaR, ES, Options, AmericanOptions


current_price = 151.03
current_date = "03/03/2023"
rf = 0.0425
div = [1]
payment_date = "03/15/2023"
dateformat = '%m/%d/%Y'


# calculate the portfolio value, pnl, delta
def get_value_pnl(portfolio, underlying_value, daysForward = 0):
    port_value = 0.0
    PnL = 0.0
    for index, row in portfolio.iterrows():
        if row["Type"] == "Stock":
            port_value += underlying_value * row["Holding"]
            PnL += (underlying_value - current_price) * row["Holding"]
        if row["Type"] == "Option":
            american_option = AmericanOptions.AmericanOption(row["OptionType"], row["ExpirationDate"], row["Strike"], current_price)
            steps, divT = american_option.get_div_T(current_date, payment_date)
            divT = [divT]
            # use real market option price to get iv, and use iv to calculate the option value with new underlying price
            iv = american_option.get_iv(current_date, rf, row["CurrentPrice"], steps = steps, div = div, divT = divT)
            
            new_date = (datetime.datetime.strptime(current_date, dateformat) + datetime.timedelta(daysForward)).strftime(dateformat)
            american_option.reset_underlying_value(underlying_value)
            ttm = american_option.get_ttm(new_date)
            new_steps, new_divT = american_option.get_div_T(new_date, payment_date)
            new_divT = [new_divT]
            value = american_option.get_value_BT(ttm, iv, rf, steps = new_steps, div = div, divT = new_divT)
            port_value += value * row["Holding"]
            PnL += (value - row["CurrentPrice"]) * row["Holding"]
    return float(port_value), float(PnL)

# calculate delta and gradient for each portfolio using current price
def get_delta_PV_gradient(portfolio, underlying_value):
    PV = 0.0
    gradient = 0.0
    # PV (sum present value of all the assets in a portfolio)
    for index, row in portfolio.iterrows():
        if row["Type"] == "Stock":
            PV += underlying_value * row["Holding"]
        if row["Type"] == "Option":
            PV += row["CurrentPrice"] * row["Holding"]
    for index, row in portfolio.iterrows():
        if row["Type"] == "Stock":
            delta = 1
            gradient += current_price / PV * row["Holding"] * delta
        if row["Type"] == "Option":
            american_option = AmericanOptions.AmericanOption(row["OptionType"], row["ExpirationDate"], row["Strike"], underlying_value)
            steps, divT = american_option.get_div_T(current_date, payment_date)
            divT = [divT]
            # use real market option price to get iv, and use iv to calculate the option value with new underlying price
            iv = american_option.get_iv(current_date, rf, row["CurrentPrice"], steps = steps, div = div, divT = divT)
            delta = american_option.get_delta(current_date, iv, rf, steps = steps, div = div, divT = divT)
            gradient += current_price / PV * row["Holding"] * delta            
    return float(PV), float(gradient)

df = pd.read_csv("problem2.csv").groupby("Portfolio")

# forward 10 days simulation
aapl_prices = pd.read_csv("DailyPrices.csv", index_col="Date")
rt = VaR.return_calculate(aapl_prices, "LOG")["AAPL"]
rt = np.array(rt[1:]).astype(np.float64)
rt = rt - np.mean(rt)
std = np.std(rt)

# simulate returns using normal distribution, 1000 simulation of forward 10 days returns to calculate 1000 simulated current prices 
n = 1000
t = 10
np.random.seed(1)
sim_rt = np.random.normal(loc=0, scale=std, size=(n, t))
sim_current_prices = []
for j in range(n):
    p = current_price
    for i in range(t):
        p *= np.exp(sim_rt[j][i])
    sim_current_prices.append(p)

sim_value_list = []
sim_PnL_list= []
for name, portfolio in df:
    d_portfolioValue = []
    d_PnL = []
    for underlyingValue in sim_current_prices:
        pt, pl= get_value_pnl(portfolio, underlyingValue, 10)
        # print(underlyingValue)
        d_portfolioValue.append(pt)
        d_PnL.append(pl)
    # print(d_portfolioValue)
    df_port_value = pd.DataFrame({name: d_portfolioValue})
    df_PnL = pd.DataFrame({name: d_PnL})
    sim_value_list.append(df_port_value)
    sim_PnL_list.append(df_PnL)
sim_port_value = pd.concat(sim_value_list, axis = 1)
sim_pnl = pd.concat(sim_PnL_list, axis = 1)
print("simulated portfolio values")
print(sim_port_value)
print("simulated portfolio pnl")
print(sim_pnl)
print()

sim_Mean = {}
sim_VaR = {}
sim_ES = {}
for name, p in sim_pnl.items():
    sim_Mean[name] = p.mean()
    var, es = ES.get_VaR_ES(p, 0.05)
    sim_VaR[name] = var
    sim_ES[name] = es
df_sim_Mean = pd.DataFrame(sim_Mean, index = ["Mean"]).T
df_sim_VaR = pd.DataFrame(sim_VaR, index = ["VaR"]).T
df_sim_ES = pd.DataFrame(sim_ES, index = ["ES"]).T
df_sim = df_sim_VaR.join(df_sim_ES.join(df_sim_Mean))
df_sim.sort_values(by = "VaR", inplace = True)
print("Calculate VaR and ES using 10-days forward simulation")
print(df_sim)

# calculate VaR and ES using Delta-Normal
sim_Mean_delta_normal = {}
sim_VaR_delta_normal = {}
sim_ES_delta_normal = {}
for name, portfolio in df:
    pv, gradient = get_delta_PV_gradient(portfolio, current_price)
    sigma = np.abs(gradient) * std
    sim_rt = st.norm.rvs(size=1000, loc=0, scale=sigma*np.sqrt(10))
    sim_pnl = pv * sim_rt
    sim_Mean_delta_normal[name] = sim_pnl.mean()
    var, es = ES.get_VaR_ES(sim_pnl, 0.05)
    sim_VaR_delta_normal[name] = var
    sim_ES_delta_normal[name] = es
df_sim_Mean_delta_normal = pd.DataFrame(sim_Mean_delta_normal, index = ["Mean"]).T
df_sim_VaR_delta_normal = pd.DataFrame(sim_VaR_delta_normal, index = ["VaR"]).T
df_sim_ES_delta_normal = pd.DataFrame(sim_ES_delta_normal, index = ["ES"]).T
df_sim = df_sim_VaR_delta_normal.join(df_sim_ES_delta_normal.join(df_sim_Mean_delta_normal))
df_sim.sort_values(by = "VaR", inplace = True)
print("Calculate VaR and ES using Delta-Normal")
print(df_sim)
    