import pandas as pd
import numpy as np
import datetime
import scipy.stats as st
from scipy.optimize import fsolve
from RiskMgmt import Options

class AmericanOption:
    def __init__(self, type, exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):
        self.type = type.lower()
        self.exp_date = exp_date
        self.K = K
        self.S0 = S0
        self.r_benefit = r_benefit
        self.r_cost = r_cost
    
    def get_ttm(self, current_date, dateformat = '%m/%d/%Y', daysForward = 0):  
        current = datetime.datetime.strptime(current_date, dateformat)
        exp = datetime.datetime.strptime(self.exp_date, dateformat)
        ttm = ((exp - current).days - daysForward) / 365
        return ttm
    
    def get_div_T(self, current_date, payment_date, dateformat = '%m/%d/%Y'):
        current = datetime.datetime.strptime(current_date, dateformat)
        exp = datetime.datetime.strptime(self.exp_date, dateformat)
        payment = datetime.datetime.strptime(payment_date, dateformat)
        days_to_exp = (exp - current).days
        days_to_pay = (payment - current).days
        return days_to_pay, days_to_exp  
    
    # calculate value for American options using Binomial Tree without dividend
    def get_value_BT_no_div(self, ttm, sigma, rf, steps):
        b = rf - self.r_benefit + self.r_cost
        dt = ttm / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        pu = (np.exp(b * dt) - d) / (u - d)
        pd = 1 - pu
        df = np.exp(-rf * dt)
        if(self.type == "call"):
            z = 1
        else:
            z = -1
            
        def get_n_nodes(n):
            return int((n + 1) * (n + 2) / 2)
        def get_idx(i, j):
            result = get_n_nodes(j - 1) + i
            return result
        
        n_nodes = get_n_nodes(steps)
        opt_values = np.zeros(n_nodes)
        
        for j in range(steps, -1, -1):
            for i in range(j, -1, -1):
                idx = get_idx(i, j)
                price = self.S0 * (u ** i) * (d **(j - i)) 
                opt_values[idx] = max([0, z * (price - self.K)])
                if(j < steps):
                    opt_values[idx] = max([opt_values[idx], df * (pu * opt_values[get_idx(i + 1, j + 1)] + pd * opt_values[get_idx(i, j + 1)])])
            
        return opt_values[0]
        
    # calculate value for American options using Binomial Tree with dividend
    def get_value_BT(self, ttm, sigma, rf, steps, div = [], divT = []):
        if len(div) == 0 or len(divT) == 0:
            return self.get_value_BT_no_div(ttm, sigma, rf, steps)
        elif divT[0] > steps:
            return self.get_value_BT_no_div(ttm, sigma, rf, steps)
        
        b = rf - self.r_benefit + self.r_cost
        dt = ttm / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        pu = (np.exp(b * dt) - d) / (u - d)
        pd = 1 - pu
        df = np.exp(-rf * dt)
        if(self.type == "call"):
            z = 1
        else:
            z = -1
            
        def get_n_nodes(n):
            return int((n + 1) * (n + 2) / 2)
        def get_idx(i, j):
            result = get_n_nodes(j - 1) + i
            return result
        
        n_nodes = get_n_nodes(divT[0])
        opt_values = np.zeros(n_nodes)
        
        for j in range(divT[0], -1, -1):
            for i in range(j, -1, -1):
                idx = get_idx(i, j)
                price = self.S0 * (u ** i) * (d **(j - i)) 
                if(j < divT[0]):
                    opt_values[idx] = max([0, z * (price - self.K)])
                    opt_values[idx] = max([opt_values[idx], df * (pu * opt_values[get_idx(i + 1, j + 1)] + pd * opt_values[get_idx(i, j + 1)])])
                else:
                    new_opt = AmericanOption(self.type, self.exp_date, self.K, price - div[0])
                    value_new = new_opt.get_value_BT(ttm - divT[0] * dt, sigma, rf, steps - divT[0], div[1:], divT[1:])
                    value = max([0, z * (price - self.K)])
                    opt_values[idx] = max([value_new, value])
        
        return opt_values[0]
        
    def reset_underlying_value(self, underlying_value):
        self.S0 = underlying_value
    
    # get option implied volatility for American Options, using value of real market price of option
    def get_iv(self, current_date, rf, price, steps, div = [], divT = []):
        
        option = Options.Option(self.type, self.exp_date, self.K, self.S0)
        ttm = option.get_T(current_date)
        #solve this function for zero to get implied volatility
        def iv_helper(vol): 
            result = price - self.get_value_BT(ttm, vol, rf, steps, div, divT) - 1e-4
            return result
        iv = fsolve(iv_helper, 0.3)
        return iv
    
    # calculate delta using closed form and finite difference(central, forward, backword difference)
    def get_delta(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        d = 1e-8 # a very small change on the denominator
        option = Options.Option(self.type, self.exp_date, self.K, self.S0)
        ttm = option.get_T(current_date)
        if how == "CENTRAL": # central difference
            opt_up = AmericanOption(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
            v_up = opt_up.get_value_BT(ttm, sigma, rf, steps, div, divT)
            opt_down = AmericanOption(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
            v_down = opt_down.get_value_BT(ttm, sigma, rf, steps, div, divT)
            delta = (v_up - v_down) / (2 * d)
        elif how == "FORWARD": # forward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            opt_up = AmericanOption(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
            v_up = opt_up.get_value_BT(ttm, sigma, rf, steps, div, divT)
            delta = (v_up - v) / d
        elif how == "BACKWARD": # backward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            opt_down = AmericanOption(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
            v_down = opt_down.get_value_BT(ttm, sigma, rf, steps, div, divT)
            delta = (v - v_down) / d
        return delta

    def get_gamma(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        d = 1e-7 # a very small change on the denominator
        if how == "CENTRAL": # central difference
            opt_up = AmericanOption(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
            delta_up = opt_up.get_delta(current_date, sigma, rf, steps, div, divT)
            opt_down = AmericanOption(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
            delta_down = opt_down.get_delta(current_date, sigma, rf, steps, div, divT)
            gamma = (delta_up - delta_down) / (2 * d)
        elif how == "FORWARD": # forward difference
            delta = self.get_delta(current_date, sigma, rf, steps, div, divT)
            opt_up = AmericanOption(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
            delta_up = opt_up.get_delta(current_date, sigma, rf, steps, div, divT)
            gamma = (delta_up - delta) / d
        elif how == "BACKWARD": # backward difference
            delta = self.get_delta(current_date, sigma, rf, steps, div, divT)
            opt_down = AmericanOption(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
            delta_down = opt_down.get_delta(current_date, sigma, rf, steps, div, divT)
            gamma = (delta - delta_down) / d
        return gamma
    
    def get_vega(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        d = 1e-8 # a very small change on the denominator
        option = Options.Option(self.type, self.exp_date, self.K, self.S0)
        ttm = option.get_T(current_date)
        if how == "CENTRAL": # central difference
            v_up = self.get_value_BT(ttm, sigma + d, rf, steps, div, divT)
            v_down = self.get_value_BT(ttm, sigma - d, rf, steps, div, divT)
            vega = (v_up - v_down) / (2 * d)
        elif how == "FORWARD": # forward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            v_up = self.get_value_BT(ttm, sigma + d, rf, steps, div, divT)
            vega = (v_up - v) / d
        elif how == "BACKWARD": # backward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            v_down = self.get_value_BT(ttm, sigma - d, rf, steps, div, divT)
            vega = (v - v_down) / d
        return vega

    def get_theta(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        d = 1e-8 # a very small change on the denominator
        option = Options.Option(self.type, self.exp_date, self.K, self.S0)
        ttm = option.get_T(current_date)
        if how == "CENTRAL": # central difference
            v_up = self.get_value_BT(ttm - d, sigma, rf, steps, div, divT)
            v_down = self.get_value_BT(ttm + d, sigma, rf, steps, div, divT)
            theta = (v_up - v_down) / (2 * d)
        elif how == "FORWARD": # forward difference, longer T
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            v_up = self.get_value_BT(ttm - d, sigma, rf, steps, div, divT)
            theta = (v_up - v) / d
        elif how == "BACKWARD": # backward difference, shorter T
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            v_down = self.get_value_BT(ttm + d, sigma, rf, steps, div, divT)
            theta = (v - v_down) / d
        return -theta
    
    def get_rho(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        d = 1e-8 # a very small change on the denominator
        option = Options.Option(self.type, self.exp_date, self.K, self.S0)
        ttm = option.get_T(current_date)
        if how == "CENTRAL": # central difference
            v_up = self.get_value_BT(ttm, sigma, rf + d, steps, div, divT)
            v_down = self.get_value_BT(ttm, sigma, rf - d, steps, div, divT)
            rho = (v_up - v_down) / (2 * d)
        elif how == "FORWARD": # forward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            v_up = self.get_value_BT(ttm, sigma, rf + d, steps, div, divT)
            rho = (v_up - v) / d
        elif how == "BACKWARD": # backward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            v_down = self.get_value_BT(ttm, sigma, rf - d, steps, div, divT)
            rho = (v - v_down) / d
        return rho
    
    def get_carry_rho(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        d = 1e-8 # a very small change on the denominator
        option = Options.Option(self.type, self.exp_date, self.K, self.S0)
        ttm = option.get_T(current_date)
        if how == "CENTRAL": # central difference
            opt_up = AmericanOption(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit - d, r_cost = self.r_cost, dateformat = dateformat)
            v_up = opt_up.get_value_BT(ttm, sigma, rf, steps, div, divT)
            opt_down = AmericanOption(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit + d, r_cost = self.r_cost, dateformat = dateformat)
            v_down = opt_down.get_value_BT(ttm, sigma, rf, steps, div, divT)
            carry_rho = (v_up - v_down) / (2 * d)
        elif how == "FORWARD": # forward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            opt_up = AmericanOption(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit - d, r_cost = self.r_cost, dateformat = dateformat)
            v_up = opt_up.get_value_BT(ttm, sigma, rf, steps, div, divT)
            carry_rho = (v_up - v) / d
        elif how == "BACKWARD": # backward difference
            v = self.get_value_BT(ttm, sigma, rf, steps, div, divT)
            opt_down = AmericanOption(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit + d, r_cost = self.r_cost, dateformat = dateformat)
            v_down = opt_down.get_value_BT(ttm, sigma, rf, steps, div, divT)
            carry_rho = (v - v_down) / d
        return carry_rho
    
    def get_greeks(self, current_date, sigma, rf, steps, div = [], divT = [], dateformat = '%m/%d/%Y', how = "CENTRAL"):
        greeks = {}
        greeks["delta"] = self.get_delta(current_date, sigma, rf, steps, div, divT, dateformat, how)
        greeks["gamma"] = self.get_gamma(current_date, sigma, rf, steps, div, divT, dateformat, how)
        greeks["theta"] = self.get_theta(current_date, sigma, rf, steps, div, divT, dateformat, how)
        greeks["vega"] = self.get_vega(current_date, sigma, rf, steps, div, divT, dateformat, how)
        greeks["rho"] = self.get_rho(current_date, sigma, rf, steps, div, divT, dateformat, how)
        greeks["carry_rho"] = self.get_carry_rho(current_date, sigma, rf, steps, div, divT, dateformat, how)
        return greeks