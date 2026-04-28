import numpy as np
import random

def jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma, (1, x.shape[1]))
    return x * factor

def window_slice(x, reduce_ratio=0.8):
    T = len(x)
    start = np.random.randint(0, int(T * (1 - reduce_ratio)))
    end = start + int(T * reduce_ratio)
    sliced = x[start:end]
    return np.stack([np.interp(np.linspace(0, 1, T),
        np.linspace(0, 1, len(sliced)), sliced[:, i]) for i in range(x.shape[1])], axis=1)

def augment(x):
    return random.choice([jitter, scaling, window_slice])(x)