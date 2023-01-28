from scipy.stats import skew, kurtosis, ttest_1samp
import numpy as np

    
def unbiased_hypo_test():
    
    # test whether the biased skewness and kurtosis calculation of scipy.stats is biased
    # sample skewness and kurtosis 100 times
    skewness_list = []
    unbias_skewness_list = []
    kurtosis_list = []
    unbias_kurtosis_list = []
    np.random.seed(1)
    for _ in range(1000):
        rand_data = np.random.normal(loc = 0, scale = 1, size = 100)
        skewness = skew(rand_data, axis = 0, bias = True)
        skewness_list.append(skewness)
        unbias_skewness = skew(rand_data, axis = 0, bias = False)
        unbias_skewness_list.append(unbias_skewness)
        # Let fisher = True to get the excess kurtosis directly, bias = True to get biased kurtosis
        excess_kurt = kurtosis(rand_data, axis = 0, fisher = True, bias = True)
        kurtosis_list.append(excess_kurt)
        unbias_excess_kurt = kurtosis(rand_data, axis = 0, fisher = True, bias = False)
        unbias_kurtosis_list.append(unbias_excess_kurt)

    # one-sample T-test, two tailed test by default
    t_stat_skew, p_value_skew = ttest_1samp(skewness_list, popmean = 0)
    t_stat_skew_unbias, p_value_skew_unbias = ttest_1samp(unbias_skewness_list, popmean = 0)
    t_stat_kurt, p_value_kurt = ttest_1samp(kurtosis_list, popmean = 0)
    t_stat_kurt_unbias, p_value_kurt_unbias = ttest_1samp(kurtosis_list, popmean = 0)
    
    # print result
    print("Test whether the skew function from scipy.stats library is biased when we set biased = True.")
    print("H0: The function is unbiased. μ_0 = 0")
    print("T-statistic value: ", t_stat_skew)  
    print("P-Value: ", p_value_skew)
    print()
    print("Test whether the skew function from scipy.stats library is biased when we set biased = False.")
    print("H0: The function is unbiased. μ_0 = 0")
    print("T-statistic value: ", t_stat_skew_unbias)  
    print("P-Value: ", p_value_skew_unbias)
    print()
    print("Test whether the kurtosis function from scipy.stats library is biased when we set biased = True.")
    print("H0: The function is unbiased. μ_0 = 0")
    print("T-statistic value: ", t_stat_kurt)  
    print("P-Value: ", p_value_kurt)
    print()
    print("Test whether the kurtosis function from scipy.stats library is biased when we set biased = False.")
    print("H0: The function is unbiased. μ_0 = 0")
    print("T-statistic value: ", t_stat_kurt_unbias)  
    print("P-Value: ", p_value_kurt_unbias)
    print()
    

if __name__ == "__main__":
    
    unbiased_hypo_test();