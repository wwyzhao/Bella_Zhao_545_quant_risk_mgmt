import pandas as pd
import numpy as np
import datetime
import scipy.stats as st
from scipy.optimize import fsolve

class Option:
    def __init__(self, type, exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):
        self.type = type.lower()
        self.exp_date = exp_date
        self.K = K
        self.S0 = S0
        self.r_benefit = r_benefit
        self.r_cost = r_cost
        
    # calcuate time to maturity(calendar days)
    # adding "daysForward" for changing time to maturity when simulation        
    def get_T(self, current_date, dateformat = '%m/%d/%Y', daysForward = 0):  
        current = datetime.datetime.strptime(current_date, dateformat)
        exp = datetime.datetime.strptime(self.exp_date, dateformat)
        T = ((exp - current).days - daysForward) / 365
        return T
    
    # reset underlying value to do simulation
    def reset_underlying_value(self, underlying_value):
        self.S0 = underlying_value
        
    # get option price using BSM function
    # adding "daysForward" for changing time to maturity when simulation
    def get_value_BS(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, T = 9999): 
        value = 0.0
        if T == 9999:
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
        r = rf
        b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continuous dividend/coupon rate
        d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(self.S0 / self.K) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if self.type == 'call':
            value = self.S0 * np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1) - self.K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
        elif self.type == 'put':
            value = self.K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1) - self.S0 * np.exp((b - r) * T) * st.norm.cdf(-d1, 0, 1)
        return value
    
    # get option implied volatility, using value of real market price of option
    def get_iv(self, current_date, rf, price, dateformat = '%m/%d/%Y', daysForward = 0):
        
        #solve this function for zero to get implied volatility
        def iv_helper(vol): 
            result = price - self.get_value_BS(current_date = current_date, sigma = vol, rf = rf, dateformat = dateformat, daysForward = daysForward) - 1e-4
            return result
   
        iv = fsolve(iv_helper, 0.3)
        return iv
    
    # calculate delta using closed form and finite difference(central, forward, backword difference)
    def get_delta(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if method == "GBSM":
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if self.type == "call":
                delta = np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1)
            elif self.type == "put":
                delta = np.exp((b - r) * T) * (st.norm.cdf(d1, 0, 1) - 1)
        else: # finite difference "FD"
            d = 1e-8 # a very small change on the denominator
            if how == "CENTRAL": # central difference
                opt_up = Option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                v_up = opt_up.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                opt_down = Option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                v_down = opt_down.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                delta = (v_up - v_down) / (2 * d)
            elif how == "FORWARD": # forward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                opt_up = Option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                v_up = opt_up.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                delta = (v_up - v) / d
            elif how == "BACKWARD": # backward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                opt_down = Option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                v_down = opt_down.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                delta = (v - v_down) / d
        return delta

    def get_gamma(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if method == "GBSM":
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            gamma = np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) / (self.S0 * sigma * np.sqrt(T))
        else: # finite difference "FD"
            d = 1e-8 # a very small change on the denominator
            if how == "CENTRAL": # central difference
                opt_up = Option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                delta_up = opt_up.get_delta(current_date, sigma, rf, dateformat, daysForward, "GBSM")
                opt_down = Option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                delta_down = opt_down.get_delta(current_date, sigma, rf, dateformat, daysForward, "GBSM")
                gamma = (delta_up - delta_down) / (2 * d)
            elif how == "FORWARD": # forward difference
                delta = self.get_delta(current_date, sigma, rf, dateformat, daysForward, "GBSM")
                opt_up = Option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                delta_up = opt_up.get_delta(current_date, sigma, rf, dateformat, daysForward, "GBSM")
                gamma = (delta_up - delta) / d
            elif how == "BACKWARD": # backward difference
                delta = self.get_delta(current_date, sigma, rf, dateformat, daysForward, "GBSM")
                opt_down = Option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                delta_down = opt_down.get_delta(current_date, sigma, rf, dateformat, daysForward, "GBSM")
                gamma = (delta - delta_down) / d
        return gamma
    
    def get_vega(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if method == "GBSM":
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = self.S0 * np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) * np.sqrt(T)
        else: # finite difference "FD"
            d = 1e-8 # a very small change on the denominator
            if how == "CENTRAL": # central difference
                v_up = self.get_value_BS(current_date, sigma + d, rf, dateformat, daysForward)
                v_down = self.get_value_BS(current_date, sigma - d, rf, dateformat, daysForward)
                vega = (v_up - v_down) / (2 * d)
            elif how == "FORWARD": # forward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                v_up = self.get_value_BS(current_date, sigma + d, rf, dateformat, daysForward)
                vega = (v_up - v) / d
            elif how == "BACKWARD": # backward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward)
                v_down = self.get_value_BS(current_date, sigma - d, rf, dateformat, daysForward)
                vega = (v - v_down) / d
        return vega

    def get_theta(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if method == "GBSM":
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(self.S0 / self.K) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if self.type == "call":
                theta = -(self.S0 * np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T))) - (b - r) * self.S0 * np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1) - r * self.K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
            elif self.type == "put":
                theta = -(self.S0 * np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T))) + (b - r) * self.S0 * np.exp((b - r) * T) * st.norm.cdf(-d1, 0, 1) + r * self.K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1)
        else: # finite difference "FD"
            d = 1e-8 # a very small change on the denominator
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            if how == "CENTRAL": # central difference
                v_up = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T - d)
                v_down = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T + d)
                theta = (v_up - v_down) / (2 * d)
            elif how == "FORWARD": # forward difference, shorter T
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                v_up = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T - d)
                theta = (v_up - v) / d
            elif how == "BACKWARD": # backward difference, longer T
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                v_down = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T + d)
                theta = (v - v_down) / d
        return -theta
    
    def get_rho(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if method == "GBSM":
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d2 = (np.log(self.S0 / self.K) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if self.type == "call":
                rho = T * self.K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
            elif self.type == "put":
                rho = -T * self.K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1)
        else: # finite difference "FD"
            d = 1e-8 # a very small change on the denominator
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            if how == "CENTRAL": # central difference
                v_up = self.get_value_BS(current_date, sigma, rf + d, dateformat, daysForward, T)
                v_down = self.get_value_BS(current_date, sigma, rf - d, dateformat, daysForward, T)
                rho = (v_up - v_down) / (2 * d)
            elif how == "FORWARD": # forward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                v_up = self.get_value_BS(current_date, sigma, rf + d, dateformat, daysForward, T)
                rho = (v_up - v) / d
            elif how == "BACKWARD": # backward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                v_down = self.get_value_BS(current_date, sigma, rf - d, dateformat, daysForward, T)
                rho = (v - v_down) / d
        return rho
    
    def get_carry_rho(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if method == "GBSM":
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if self.type == "call":
                carry_rho = T * self.S0 * np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1)
            elif self.type == "put":
                carry_rho = -T * self.S0 * np.exp((b - r) * T) * st.norm.cdf(-d1, 0, 1)
        else: # finite difference "FD"
            d = 1e-8 # a very small change on the denominator
            T = self.get_T(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            if how == "CENTRAL": # central difference
                opt_up = Option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit - d, r_cost = self.r_cost, dateformat = dateformat)
                v_up = opt_up.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                opt_down = Option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit + d, r_cost = self.r_cost, dateformat = dateformat)
                v_down = opt_down.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                carry_rho = (v_up - v_down) / (2 * d)
            elif how == "FORWARD": # forward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                opt_up = Option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit - d, r_cost = self.r_cost, dateformat = dateformat)
                v_up = opt_up.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                carry_rho = (v_up - v) / d
            elif how == "BACKWARD": # backward difference
                v = self.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                opt_down = Option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit + d, r_cost = self.r_cost, dateformat = dateformat)
                v_down = opt_down.get_value_BS(current_date, sigma, rf, dateformat, daysForward, T)
                carry_rho = (v - v_down) / d
        return carry_rho
    
    def get_greeks(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        greeks = {}
        greeks["delta"] = self.get_delta(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["gamma"] = self.get_gamma(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["theta"] = self.get_theta(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["vega"] = self.get_vega(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["rho"] = self.get_rho(current_date, sigma, rf, dateformat, daysForward, method,how)
        greeks["carry_rho"] = self.get_carry_rho(current_date, sigma, rf, dateformat, daysForward, method, how)
        return greeks