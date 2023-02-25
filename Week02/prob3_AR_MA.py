import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import pacf,acf

np.random.seed(1)
n = 1000
burn_in = 50
e = np.random.normal(loc = 0, scale = 0.01, size = n + burn_in)

def AR():

    # AR1
    # y_t = 1.0 + 0.5 * y_t-1 + e, e ~ N(0,0.01)
    y_t1 = np.zeros(n + burn_in)
    y_t1[0] = e[0]
    for i in range(1, n + burn_in):
        y_t1[i] = 1 + 0.5 * y_t1[i - 1] + e[i]
    y_t1 = y_t1[burn_in:]

    # AR2
    # y_t = 1 + 0.5 * y_t-1 - 0.5 * y_t-2 + e, e ~ N(0,0.01)
    y_t2 = np.zeros(n + burn_in)
    y_t2[0:2] = e[0:2]
    for i in range(2, n + burn_in):
        y_t2[i] = 1 + 0.5 * y_t2[i - 1] - 0.25 * y_t2[i - 2] + e[i]
    y_t2 = y_t2[burn_in:]

    # AR3
    # y_t = 1 + 0.5 * y_t-1 - 0.5 * y_t-2 + 0.5 * y_t-3 + e, e ~ N(0,0.01)
    y_t3 = np.zeros(n + burn_in)
    y_t3[0:3] = e[0:3]
    for i in range(3, n + burn_in):
        y_t3[i] = 1 + 0.5 * y_t3[i - 1] - 0.25 * y_t3[i - 2] + 0.125 * y_t3[i - 3] + e[i]
    y_t3 = y_t3[burn_in:]

    fig, axes = plt.subplots(1, 3, figsize=(30, 5))
    fig.suptitle('AR Processes')
    sns.lineplot(ax=axes[0], x=range(1000), y=y_t1)
    axes[0].set_title("AR(1)")
    sns.lineplot(ax=axes[1], x=range(1000), y=y_t2)
    axes[1].set_title("AR(2)")
    sns.lineplot(ax=axes[2], x=range(1000), y=y_t3)
    axes[2].set_title("AR(3)")
    plt.show()

    # ACF PACF
    nn = len(acf(y_t1))
    fig_, axes_ = plt.subplots(1, 3, figsize=(15, 5))
    fig_.suptitle('AR Processes')
    sns.lineplot(ax=axes_[0], x=range(nn), y=acf(y_t1), label="ACF")
    sns.lineplot(ax=axes_[0], x=range(nn), y=pacf(y_t1), label="PACF")
    axes[0].set_title("AR(1)")
    sns.lineplot(ax=axes_[1], x=range(nn), y=acf(y_t2), label="ACF")
    sns.lineplot(ax=axes_[1], x=range(nn), y=pacf(y_t2), label="PACF")
    axes[1].set_title("AR(2)")
    sns.lineplot(ax=axes_[2], x=range(nn), y=acf(y_t3), label="ACF")
    sns.lineplot(ax=axes_[2], x=range(nn), y=pacf(y_t3), label="PACF")
    axes[2].set_title("AR(3)")
    plt.show()


def MA():

    # MA1
    # y_t = 1.0 + 0.5 * e_t-1 + e, e ~ N(0,.01)
    y_t1 = np.zeros(n + burn_in)
    y_t1[0] = e[0]
    for i in range(1, n + burn_in):
        y_t1[i] = 1.0 + 0.5 * e[i - 1] + e[i]
    y_t1 = y_t1[burn_in:]

    # MA2
    # y_t = 1 + 0.5 * e_t-1 + 0.5 * e_t-2 + e, e ~ N(0,.01)
    y_t2 = np.zeros(n + burn_in)
    y_t2[0:2] = e[0:2]
    for i in range(2, n + burn_in):
        y_t2[i] = 1 + 0.5 * e[i - 1] + 0.5 * e[i - 2] + e[i]
    y_t2 = y_t2[burn_in:]

    # MA3
    # y_t = 1 + 0.5 * e_t-1 + 0.5 * e_t-2 + 0.5 * e_t-3 + e, e ~ N(0,.01)
    y_t3 = np.zeros(n + burn_in)
    y_t3[0:3] = e[0:3]
    for i in range(3, n + burn_in):
        y_t3[i] = 1 + 0.5 * e[i - 1] + 0.5 * e[i - 2] + 0.5 * e[i - 3] + e[i]
    y_t3 = y_t3[burn_in:]

    fig, axes = plt.subplots(1, 3, figsize=(30, 5))
    fig.suptitle('MA Processes')
    sns.lineplot(ax=axes[0], x=range(1000), y=y_t1)
    axes[0].set_title("MA(1)")
    sns.lineplot(ax=axes[1], x=range(1000), y=y_t2)
    axes[1].set_title("MA(2)")
    sns.lineplot(ax=axes[2], x=range(1000), y=y_t3)
    axes[2].set_title("MA(3)")
    plt.show()

    # ACF PACF
    nn = len(acf(y_t1))
    fig_, axes_ = plt.subplots(1, 3, figsize=(15, 5))
    fig_.suptitle('MA Processes')
    sns.lineplot(ax=axes_[0], x=range(nn), y=acf(y_t1), label="ACF")
    sns.lineplot(ax=axes_[0], x=range(nn), y=pacf(y_t1), label="PACF")
    axes[0].set_title("MA(1)")
    sns.lineplot(ax=axes_[1], x=range(nn), y=acf(y_t2), label="ACF")
    sns.lineplot(ax=axes_[1], x=range(nn), y=pacf(y_t2), label="PACF")
    axes[1].set_title("MA(2)")
    sns.lineplot(ax=axes_[2], x=range(nn), y=acf(y_t3), label="ACF")
    sns.lineplot(ax=axes_[2], x=range(nn), y=pacf(y_t3), label="PACF")
    axes[2].set_title("MA(3)")
    plt.show()

if __name__ == '__main__':

    AR()

    MA()
