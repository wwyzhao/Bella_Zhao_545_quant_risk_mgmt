import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RiskMgmt import Options, AmericanOptions

S0 = 165
K = 165
current_date = "03/13/2022"
exp_date = "04/15/2022"
rf = 0.0425
r_benefit = 0.0053
iv = 0.2

# closed form greeks for GBSM, finite difference derivative calculation(central, forward, backward)
option_c = Options.Option("call", exp_date, K, S0, r_benefit)
option_p = Options.Option("put", exp_date, K, S0, r_benefit)
ttm = option_c.get_T(current_date)

c_gbsm = option_c.get_greeks(current_date, iv, rf, method = "GBSM")
c_fd_central = option_c.get_greeks(current_date, iv, rf, method = "FD")
c_fd_forward = option_c.get_greeks(current_date, iv, rf, method = "FD", how = "FORWARD")
c_fd_backward = option_c.get_greeks(current_date, iv, rf, method = "FD", how = "BACKWARD")
p_gbsm = option_p.get_greeks(current_date, iv, rf, method = "GBSM")
p_fd_central = option_p.get_greeks(current_date, iv, rf, method = "FD")
p_fd_forward = option_p.get_greeks(current_date, iv, rf, method = "FD", how = "FORWARD")
p_fd_backward = option_p.get_greeks(current_date, iv, rf, method = "FD", how = "BACKWARD")
greeks = [pd.DataFrame(c_gbsm, index=["Call(GBSM)"]).T,
          pd.DataFrame(c_fd_central, index=["Call(FD_Central)"]).T,
          pd.DataFrame(c_fd_central, index=["Call(FD_Forward)"]).T,
          pd.DataFrame(c_fd_backward, index=["Call(FD_Backward)"]).T,
          pd.DataFrame(p_gbsm, index=["Put(GBSM)"]).T,
          pd.DataFrame(p_fd_central, index=["Put(FD_Central)"]).T,
          pd.DataFrame(p_fd_forward, index=["Put(FD_Forward)"]).T,
          pd.DataFrame(p_fd_backward, index=["Put(FD_Backward)"]).T]
df = pd.concat(greeks, axis=1)
print("Closed form greeks:")
print(df)

# American Options without discrete dividends
american_c = AmericanOptions.AmericanOption("call", exp_date, K, S0, r_benefit)
value_c = american_c.get_value_BT_no_div(ttm, iv, rf, steps=33)
greeks_c_central = american_c.get_greeks(current_date, iv, rf, steps=33, how = "CENTRAL")
american_p = AmericanOptions.AmericanOption("put", exp_date, K, S0, r_benefit)
value_p = american_p.get_value_BT_no_div(ttm, iv, rf, steps=33)
greeks_p_central = american_p.get_greeks(current_date, iv, rf, steps=33, how = "CENTRAL")
greeks = [pd.DataFrame(greeks_c_central, index=["Call(FD_Central)"]).T,
          pd.DataFrame(greeks_p_central, index=["Put(FD_Central)"]).T]
df = pd.concat(greeks, axis=1)
print("American call option value by Binomial Tree(without dividend): ", value_c)
print("American put option value by Binomial Tree(without dividend): ", value_p)
print("Finite difference derivative calculation greeks(without dividend):")
print(df)

# American Options with discrete dividends
american_c = AmericanOptions.AmericanOption("call", exp_date, K, S0, r_benefit)
value_c = american_c.get_value_BT(ttm, iv, rf, steps=33, div=[0.88], divT=[29])
greeks_c_central = american_c.get_greeks(current_date, iv, rf, steps=33, div=[0.88], divT=[29], how = "CENTRAL")
american_p = AmericanOptions.AmericanOption("put", exp_date, K, S0, r_benefit)
value_p = american_p.get_value_BT(ttm, iv, rf, steps=33, div=[0.88], divT=[29])
greeks_p_central = american_p.get_greeks(current_date, iv, rf, steps=33, div=[0.88], divT=[29], how = "CENTRAL")
greeks = [pd.DataFrame(greeks_c_central, index=["Call(FD_Central)"]).T,
          pd.DataFrame(greeks_p_central, index=["Put(FD_Central)"]).T]
df = pd.concat(greeks, axis=1)
print("American call option value by Binomial Tree(with dividend): ", value_c)
print("American put option value by Binomial Tree(with dividend): ", value_p)
print("Finite difference derivative calculation greeks(with dividend):")
print(df)

# sensitivity of the put and call to a change in the dividend amount
dividend = np.linspace(0,1)
values_c = []
values_p = []
for div in dividend:
    values_c.append(american_c.get_value_BT(ttm, iv, rf, steps=33, div=[div], divT=[29]))
    values_p.append(american_p.get_value_BT(ttm, iv, rf, steps=33, div=[div], divT=[29]))
plt.cla()
plt.plot(dividend, values_c, label="American Call")
plt.ylabel("Option Value")
plt.xlabel("Dividend Amount")
plt.legend()
plt.suptitle("American Call Sensitivity Comparision")
plt.savefig("plots/problem1_sensitivity_dividend_call.png")

plt.cla()
plt.plot(dividend, values_p, label="American Put")
plt.ylabel("Option Value")
plt.xlabel("Dividend Amount")
plt.legend()
plt.suptitle("American Put Sensitivity Comparision")
plt.savefig("plots/problem1_sensitivity_dividend_put.png")