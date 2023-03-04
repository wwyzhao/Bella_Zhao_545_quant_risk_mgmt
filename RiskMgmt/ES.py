# module of calculation of ES and absolute VaR
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

# absolute value of VaR for original data
def get_VaR_ES(rt, alpha):
    """Calculate VaR using original data

    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: absolute VaR and ES (negative values)
    """
    mu = np.mean(rt)
    sigma = np.std(rt)
    VaR = np.percentile(rt, alpha * 100)
    temp = rt[rt <= VaR].dropna()
    ES = np.mean(temp)
    
    plt.cla()
    sns.histplot(rt, kde=True, color="blue", label="original data")
    plt.axvline(VaR, color='r', label='VaR')
    plt.axvline(ES, color='g', label='ES')
    plt.title("VaR_ES")
    plt.savefig("plots/problem3_VaR_ES.png")
    
    return -VaR, -ES
    
# absolute value of VaR for fitted normal distribution
def get_VaR_ES_normal(rt, alpha):
    """Calculate VaR using fitted normal distribution

    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: absolute VaR and ES (negative values)
    """
    mu = np.mean(rt)
    sigma = np.std(rt)
    np.random.seed(1)
    fit_normal = np.random.normal(loc=mu, scale=sigma, size=len(rt))
    VaR = st.norm.ppf(alpha, loc=mu, scale=sigma)
    temp = fit_normal[fit_normal <= VaR]
    ES = np.mean(temp)
    # ES_Normal = - (-mu + sigma * (st.norm.pdf(st.norm.ppf(alpha)))/alpha)

    plt.cla()
    sns.histplot(rt, kde=True, color="blue", label="original data")
    sns.histplot(fit_normal, kde=True, color="orange", label="fitted_normal")
    plt.axvline(VaR, color='r', label='VaR_Normal')
    plt.axvline(ES, color='g', label='ES_Normal')
    plt.title("VaR_ES_normal")
    plt.savefig("plots/problem3_VaR_ES_normal.png")

    return -VaR, -ES

# absolute value of VaR for fitted T distribution
def get_VaR_ES_T(rt, alpha):
    """Calculate VaR using fitted T distribution

    Args:
        rt (np.array): returns
        alpha (float): confidence

    Returns:
        float: absolute VaR and ES (negative values)
    """
    mu = np.mean(rt)
    sigma = np.std(rt)
    np.random.seed(1)
    df, loc, scale = st.t.fit(rt, loc=mu, scale=sigma)
    fit_t = st.t.rvs(df = df, loc = loc, scale = scale, size = len(rt))
    VaR = st.t.ppf(alpha, df=df, loc=loc, scale=scale)
    temp = fit_t[fit_t <= VaR]
    ES = np.mean(temp)
    
    plt.cla()
    sns.histplot(rt, kde=True, color="blue", label="original data")
    sns.histplot(fit_t, kde=True, color="orange", label="fitted T")
    plt.axvline(VaR, color='r', label='VaR_T')
    plt.axvline(ES, color='g', label='ES_T')
    plt.title("VaR_ES_T")
    plt.savefig("plots/problem3_VaR_ES_T.png")
    
    return -VaR, -ES

def get_VaR_ES_historic(rt, alpha):

    times = len(rt)
    np.random.seed(1)
    his_distribution = np.random.choice(rt, size=times, replace=True)
    VaR = np.percentile(his_distribution, alpha * 100)
    temp = his_distribution[his_distribution <= VaR]
    ES = np.mean(temp)
    
    plt.cla()
    sns.histplot(rt, kde=True, color="blue", label="original data")
    sns.histplot(his_distribution, kde=True, color="orange", label="fitted_historical")
    plt.axvline(VaR, color='r', label='VaR_Normal')
    plt.axvline(ES, color='g', label='ES_Normal')
    plt.title("VaR_ES_historical")
    plt.savefig("plots/problem3_VaR_ES_historical.png")
    
    return -VaR, -ES