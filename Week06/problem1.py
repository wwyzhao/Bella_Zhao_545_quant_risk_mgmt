import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import numpy as np
import matplotlib.pyplot as plt
from RiskMgmt import Options


S0 = 165
current_date = "03/03/2023"
exp_date = "03/17/2023"
rf = 0.0425
r_benefit = 0.0053

# test1
K_call = 170 # assumption of the call option strike price
K_put = 160 # assumption of the put option strike price
option_c = Options.Option("call", exp_date, K_call, S0, r_benefit)
option_p = Options.Option("put", exp_date, K_put, S0, r_benefit)
T = option_c.get_T(current_date)
print(f"Time to maturity: {T:1.3f} years")
c_list = []
p_list = []
iv_list = []
for iv in np.arange(0.1, 0.81, 0.01):
    c = option_c.get_value_BS(current_date, iv, rf)
    p = option_p.get_value_BS(current_date, iv, rf)
    iv_list.append(iv)
    c_list.append(c)
    p_list.append(p)

# test2
K_call1 = 160 # assumption of the call option strike price
K_put1 = 170 # assumption of the put option strike price
option_c = Options.Option("call", exp_date, K_call1, S0, r_benefit)
option_p = Options.Option("put", exp_date, K_put1, S0, r_benefit)
c_list1 = []
p_list1 = []
for iv in np.arange(0.1, 0.81, 0.01):
    c = option_c.get_value_BS(current_date, iv, rf)
    p = option_p.get_value_BS(current_date, iv, rf)
    c_list1.append(c)
    p_list1.append(p)
    
plt.cla()
plt.plot(iv_list, c_list, label = "call(K = " + str(K_call) + ")")
plt.plot(iv_list, p_list, color = 'orange', label = "put(K = " + str(K_put) + ")")
plt.plot(iv_list, c_list1, color = 'r', label = "call(K = " + str(K_call1) + ")")
plt.plot(iv_list, p_list1, color = 'g', label = "put(K = " + str(K_put1) + ")")
plt.xlabel("Implied Volatility")
plt.ylabel("BS Value")
plt.legend()
plt.title("Option Value (S0 = 165)")
plt.savefig("plots/problem1_option_value.png")
    


