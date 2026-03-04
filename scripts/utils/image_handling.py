import numpy as np

def findrms(mIn, maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS
    """
    try:
        m = mIn[np.abs(mIn) > maskSup]
        rmsold = np.std(m)
        diff = 1e-1
        cut = 3.
        bins = np.arange(np.min(m), np.max(m), (np.max(m) - np.min(m)) / 30.)
        med = np.median(m)
        for i in range(10):
            ind = np.where(np.abs(m - med) < rmsold * cut)[0]
            rms = np.std(m[ind])
            if np.abs((rms - rmsold) / rmsold) < diff: break
            rmsold = rms
        return rms
    except ValueError:
        return np.nan


def clipped_median(data, sigma_clip=3.0, max_iter=5):
    """
    Calculate the clipped median of an 1D array.
    """
    data = data[np.isfinite(data)]  # Remove NaN and Inf values
    for _ in range(max_iter):
        median = np.median(data)
        std_dev = np.std(data)
        mask = np.abs(data - median) < sigma_clip * std_dev
        new_data = data[mask]
        if len(new_data) == len(data):
            break
        data = new_data
    return np.median(data)
