import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
from numpy.linalg import norm
from RiskMgmt import CovEstimation, NonPSDFix, Simulation, VaR, ES

# testing the covariance estimation module
def test_covariance():
    
    df = pd.read_csv("DailyReturn.csv")
    cor_P, var_P, cov_P = CovEstimation.pearson_cor_var_cov(df)
    cor_w, var_w, cov_w = CovEstimation.exp_w_cor_var_cov(df, 0.97)

# testing the psd module
def test_psd():
    # get a non psd matrix
    sigma = NonPSDFix.get_nonpsd(5)

    if NonPSDFix.is_PSD(sigma): 
        print("Original matrix is psd")
    else:
        print("Original matrix is not psd")
    
    if NonPSDFix.is_PSD(NonPSDFix.near_psd(sigma)): 
        print("Near PSD matrix is psd")
    else:
        print("Near PSD matrix is not psd")
    
    if NonPSDFix.is_PSD(NonPSDFix.higham_psd(sigma)): 
        print("Higham PSD matrix is psd")
    else:
        print("Higham PSD matrix is not psd")

# testing the simulation module
def test_simulation():

    df = pd.read_csv("DailyReturn.csv")
    cor_P, var_P, cov_P = CovEstimation.pearson_cor_var_cov(df)
    nsim = 25000
    
    # cov_PSD = NonPSDFix.near_psd(cov_P)
    # df = NonPSDFix.get_psd(5)
    # df = pd.DataFrame(df)
    # cor_P, var_P, cov_P = CovEstimation.pearson_cor_var_cov(df)
    s_direct = Simulation.simulate_direct(cov_P, nsim)
    s_pca = Simulation.simulate_pca(np.array(cov_P), nsim, 0.75)

    cov_direct = np.cov(s_direct)
    diff_direct = norm(cov_direct - cov_P, 'fro')
    cov_pca = np.cov(s_pca)
    diff_pca = norm(cov_pca - cov_P, 'fro')

    print("norm of direct simulation: ", diff_direct)
    print("norm of pca simulation: ", diff_pca)

# testing the VaR module
def test_var():
    
    #use problem1's data for testing
    data = pd.read_csv("problem1.csv")
    data = np.array(data["x"].copy())
    data = data - np.mean(data)
    
    VaR_Normal = VaR.get_VaR_normal(data, 0.05) 
    print("VaR_Normal: ", VaR_Normal)
    
    VaR_Normal_w = VaR.get_VaR_exp_w(data, 0.05, 0.97) 
    print("VaR_Normal_exp_weighted: ", VaR_Normal_w)
    
    VaR_t_mle = VaR.get_VaR_MLE_T(data, 0.05)  
    print("VaR_t_mle: ", VaR_t_mle)
    
    VaR_AR_1 = VaR.get_VaR_AR_1(data, 0.05)  
    print("VaR_AR_1: ", VaR_AR_1)
    
    VaR_hist = VaR.get_VaR_historic(data, 0.05) 
    print("VaR_hist: ", VaR_hist)
    
#testing the ES module
def test_es():
    
    data = pd.read_csv("problem1.csv")
    data = np.array(data["x"].copy())
    VaR_normal, ES_Normal = ES.get_VaR_ES_normal(data, 0.05) 
    print("ES_Normal: ", ES_Normal)
    
    VaR_T, ES_T = ES.get_VaR_ES_T(data, 0.05)
    print("ES_T: ", ES_T)

if __name__=='__main__':
    test_covariance()
    test_psd()
    test_simulation()
    test_var()
    test_es()
    
    