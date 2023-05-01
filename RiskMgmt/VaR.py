#module calculation of return and VaR (relative values, mu = 0)
import pandas as pd
import numpy as np
import statsmodels.tsa.arima.model as sm
import scipy.stats as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

def return_calculate(data, method="ARITH_RT"):
    """three methods of return

    Args:
        data (pandas.df): dataframe of price matrix
        method (string): calculation methods

    Returns:
        pandas.df: return matrix
    """

    rt = pd.DataFrame(index=data.index, columns=data.columns)

    # Classical Brownian Motion
    if method == "BM":
        rt = data - data.shift(1).fillna(0)
    # Arithmetic Return System
    elif method == "ARITH_RT":
        rt = data / data.shift(1).fillna(1) - 1
    # Geometric Brownian Motions
    else:
        rt = np.log(data.astype(np.float64) / data.shift(1).fillna(1).astype(np.float64))

    # The first row of data has no return
    rt.iloc[0,:] = "NaN"
    return rt

def get_VaR_normal(rt, alpha):
    """Calculate VaR using a normal distribution

    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: relative VaR
    """
    mu = 0
    sigma = np.std(rt)
    VaR = -st.norm.ppf(alpha, loc=mu, scale=sigma) # mu = 0, symmetric

    # plt.cla()
    # sns.histplot(rt, kde=True, color="blue")
    # np.random.seed(1)
    # x = np.random.normal(loc=0, scale=sigma, size=len(rt))
    # sns.histplot(x, kde=True, color="orange")
    # plt.title("VaR_normal")
    # plt.savefig("plots/problem2_VaR_normal.png")

    return VaR

# Calculate VaR using a normal distribution with an Exponentially Weighted variance
def get_VaR_exp_w(rt, alpha, lambd):
    """Calculate VaR using a normal distribution with an Exponentially Weighted variance

    Args:
        rt (np.array): returns
        alpha (float): confidence
        lambd (float): exp parameter

    Returns:
        float: relative VaR
    """
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

    # plt.cla()
    # sns.histplot(rt, kde=True, color="blue")
    # np.random.seed(1)
    # x = np.random.normal(loc=0, scale=sigma, size=len(rt))
    # sns.histplot(x, kde=True, color="orange")
    # plt.title("VaR_exp_w")
    # plt.savefig("plots/problem2_VaR_exp_w.png")

    return VaR

# Calculate VaR using a MLE fitted T distribution 
def MLE_T(parameters, x):
    mu, std, degree = parameters
    LL = np.sum(st.t.logpdf(x, loc=mu, scale=std, df=degree))
    return -LL

def MLE_T_Simulation(rt):
    mu = 0
    sigma = np.std(rt)
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - 0}) # constraints: standard deviation >= 0 
    mle_t = minimize(MLE_T, np.array([mu, sigma, rt.shape[0]-1]), args=rt, constraints = cons)
    return mle_t.x

def get_VaR_MLE_T(rt, alpha):
    """Calculate VaR using a MLE fitted T distribution 
     
    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: relative VaR
    """
    mle_t_distribution = MLE_T_Simulation(rt)
    # print(mle_t_distribution)
    df = mle_t_distribution[2]
    loc = mle_t_distribution[0]
    scale = mle_t_distribution[1]
    VaR = -st.t.ppf(alpha, df=df, loc=loc, scale=scale)
    
    # plt.cla()
    # sns.histplot(rt, kde=True, color="blue")
    # np.random.seed(1)
    # x = np.linspace(st.t.ppf(0.001, df, loc, scale), st.t.ppf(0.999, df, loc, scale), 20000)
    # y = st.t.pdf(x, df, loc, scale)
    # plt.plot(x, y)
    # sns.lineplot(x=x, y=y, color="orange")
    # plt.title("VaR_MLE_T")
    # plt.savefig("plots/problem2_VaR_MLE_T.png")

    return VaR

# Calculate VaR using a fitted AR(1) model
def get_VaR_AR_1(rt, alpha):
    """Calculate VaR using a fitted AR(1) model
     
    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: relative VaR
    """
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

    # plt.cla()
    # sns.histplot(rt, kde=True, color="blue")
    # np.random.seed(1)
    # sns.histplot(sim_rt, kde=True, color="orange")
    # plt.title("VaR_AR_1")
    # plt.savefig("plots/problem2_VaR_AR_1.png")

    return VaR

# Calculate VaR using a Historic Simulation
def get_VaR_historic(rt, alpha):
    """Calculate VaR using a Historic Simulation
     
    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: relative VaR
    """
    times = len(rt)
    np.random.seed(1)
    his_distribution = np.random.choice(rt, size=times, replace=True)
    VaR = -np.percentile(his_distribution, alpha * 100)

    # plt.cla()
    # sns.histplot(rt, kde=True, color="blue")
    # np.random.seed(1)
    # sns.histplot(his_distribution, kde=True, color="orange")
    # plt.title("VaR_historic")
    # plt.savefig("plots/problem2_VaR_historic.png")

    return VaR