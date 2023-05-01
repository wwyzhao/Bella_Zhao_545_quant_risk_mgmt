from scipy.stats import skew, kurtosis
import numpy as np

def get_four_elements(rt):
    avg = np.mean(rt)
    std = np.std(rt)
    skewness =  skew(rt, axis = 0)
    excess_kurt = kurtosis(rand_data, axis = 0, fisher = True)
    print("Mean: ", avg)
    print("StDev: ", std)
    print("Skew: ", skewness)
    print("Kurt: ", excess_kurt)
    return avg, std, skewness, excess_kurt
