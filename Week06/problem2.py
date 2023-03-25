import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import matplotlib.pyplot as plt
from RiskMgmt import Options


S0 = 151.03
current_date = "03/03/2023"
exp_date = "03/17/2023"
rf = 0.0425
r_benefit = 0.0053

c_strike = []
p_strike = []
c_list = []
p_list = []
df = pd.read_csv("AAPL_Options.csv")
for i in range(len(df)):
    aapl_option = Options.Option(type=df["Type"].loc[i], exp_date=exp_date, K=df["Strike"].loc[i], S0=S0, r_benefit=r_benefit)
    if aapl_option.type == "call":
        c_strike.append(aapl_option.K)
        c_list.append(aapl_option.get_iv(current_date, rf, df["Last Price"].loc[i]))
    else:
        p_strike.append(aapl_option.K)
        p_list.append(aapl_option.get_iv(current_date, rf, df["Last Price"].loc[i]))

plt.cla()
plt.plot(c_strike, c_list)
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility -- Call AAPL Options")
plt.savefig("plots/problem2_iv_call.png")

plt.cla()
plt.plot(p_strike, p_list)
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility -- Put AAPL Options")
plt.savefig("plots/probelm2_iv_put.png")