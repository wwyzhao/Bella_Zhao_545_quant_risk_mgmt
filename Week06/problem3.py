import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as sm
from RiskMgmt import VaR, ES, Options


current_price = 151.03
current_date = "03/03/2023"
rf = 0.0425
r_benefit = 0.0053

# calculate the portfolio value
def get_port_value(portfolio, underlying_value, daysForward = 0):
    port_value = 0.0
    PnL = 0.0
    for index, row in portfolio.iterrows():
        if row["Type"] == "Stock":
            port_value += underlying_value * row["Holding"]
            PnL += (underlying_value - current_price) * row["Holding"]
        if row["Type"] == "Option":
            option = Options.Option(row["OptionType"], row["ExpirationDate"], row["Strike"], current_price, r_benefit)
            # use real market option price to get iv, and use iv to calculate the option value with new underlying price
            iv = option.get_iv(current_date=current_date, rf=rf, price=row["CurrentPrice"], daysForward=daysForward)
            option.reset_underlying_value(underlying_value)
            value = option.get_value_BS(current_date=current_date, sigma=iv, rf=rf, daysForward=daysForward)
            port_value += value * row["Holding"]
            PnL += (value - row["CurrentPrice"]) * row["Holding"]
    return float(port_value), float(PnL)

# plot portfolio value and pnl
df = pd.read_csv("problem3.csv").groupby("Portfolio")

value_list = []
PnL_list= []
for name, portfolio in df:
    d_portfolioValue = {}
    d_PnL = {}
    for underlyingValue in range(130, 170):
        d_portfolioValue[underlyingValue], d_PnL[underlyingValue] = get_port_value(portfolio, underlyingValue)
    df_port_value = pd.DataFrame(d_portfolioValue, index = [name]).T
    df_PnL = pd.DataFrame(d_PnL, index = [name]).T
    value_list.append(df_port_value)
    PnL_list.append(df_PnL)
    
port_value = pd.concat(value_list, axis = 1)
port_value.plot(figsize = (10, 10),
                title = "Portfolio Value",
                xlabel = "Underlying Value",
                ylabel = "Portfolio Value",
                legend = 1)
plt.savefig("plots/problem3_port_value.png")
pnl = pd.concat(PnL_list, axis = 1)
pnl.plot(figsize = (10, 10),
                title = "Portfolio PnL",
                xlabel = "Underlying Value",
                ylabel = "Portfolio PnL",
                legend = 1)
plt.savefig("plots/problem3_pnl.png")


# forward 10 days simulation
aapl_prices = pd.read_csv("DailyPrices.csv", index_col="Date")
rt = VaR.return_calculate(aapl_prices, "LOG")["AAPL"]
rt = np.array(rt[1:]).astype(np.float64)
rt = rt - np.mean(rt)

# fit AR(1) model to return
model = sm.ARIMA(rt, order=(1, 0, 0)).fit()
# autoregressive coefficient and error variance
ar_coef = model.arparams[0]
err_var = model.mse

# simulate returns using AR(1) model, 1000 simulation of forward 10 days returns to calculate 1000 simulated current prices 
n = 1000
t = 10
np.random.seed(1)
r = np.random.normal(loc=0, scale=1, size=(n, t))
sim_current_prices = []
for j in range(n):
    sim_rt = np.zeros(t+1)
    sim_rt[0] = rt[-1]
    p = current_price
    for i in range(1, t+1):
        sim_rt[i] = ar_coef * sim_rt[i - 1] + np.sqrt(err_var) * r[j][i - 1]
        p *= np.exp(sim_rt[i])
    sim_current_prices.append(p)
    
sim_value_list = []
sim_PnL_list= []
for name, portfolio in df:
    d_portfolioValue = []
    d_PnL = []
    for underlyingValue in sim_current_prices:
        pt, pl = get_port_value(portfolio, underlyingValue, 10)
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
print(df_sim)



