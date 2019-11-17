
__title__ = 'robustad'
__version__ = '0.1.0'

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import find_peaks


class RobustAnomalyDetector:
    
    def __init__(self, alpha = 6, seasonality_lag = 0, log_transform = False):
        """
        Class for training a model and predicting anomalies in given data.

        Parameters
        ----------
        alpha : float, optional (default 6)
            Model hyperparameter for optimal anomaly-bounds definition.
            The larger value of alpha the less sensitive will be end model to data outliers.
            It is not recommended to set alpha less than 2 due to defition of median absolute deviation
        seasonality_lag : int, optional (default 0)
            The seasonality lag value. 
            Default zero value means there is no evidence of seasonality in given data
        log_transform : bool, optional (default False)
            Either log-transform data if there is evidence for multiplicative modelling
        """
        
        self.alpha = alpha
        self.seasonality_lag = seasonality_lag
        self.log_transform = log_transform
        self.bound = 0
        self.peak_labels = []
        self.collaples_labels = []
        
    def _MAD(self, X):
        """
        Median absolute deviation (MAD) calculation
        """
        
        X_hat = np.median(X)
        return np.median(np.abs(X - X_hat))
    
    def _determine_lag(self, X):
        """
        Function to obtain optimal seasonality parameter from given data
        """
        
        if self.seasonality_lag == 1:
            raise ValueError("seasonality_lag cannot be less or equal to 1")
        
        if self.seasonality_lag == 0:
            acf = sm.tsa.acf(X.values.squeeze(), nlags = len(X) - 1, fft=False)
            peaks, _ = find_peaks(acf, height = 0)
        
            if len(peaks) == 0:
                warnings.warn("seasonality_lag cannot be estimated.\
                    Please ensure there is no seasonality either enforce seasonality in object argument")
            elif (len(peaks[peaks < len(X) - 15]) + 1) < (len(X) // peaks[0]):
                warnings.warn("Seasonality seems to have weak evidence.\
                    Please ensure there is no seasonality either enforce seasonality in object argument")
            elif np.sum(peaks[: 3] == [peaks[0], peaks[0] * 2, peaks[0] * 3]) == 3 and acf[peaks[0]] > 0.5:
                warnings.warn("Seasonality seems to have strong evidence \
                              with seasonal component = " + str(peaks[0]))
                self.seasonality_lag = peaks[0]
            else:
                self.seasonality_lag = peaks[0]
            
    def _pandas_convertion(self, X):
        """
        Trying to convert original array to pandas Series representation.
        Therefore it is not recommended to fit abnormal arrays that pandas cannot process
        """
        
        return pd.Series(list(X)).apply(np.float64)
    
    def _check_array(self, X):
        """
        Temporal limitation for array length
        """
        
        if len(X) < 30:
            raise ValueError('Length of X must be greater than 30')
    
    def fit(self, X):
        """
        Fit the model with given data
        """
        
        X = self._pandas_convertion(X)
        
        self._check_array(X)
        
        self._determine_lag(X)
        
        if self.log_transform == True:
            X = X.apply(lambda x: np.log(x + 1 + np.min(X)) if np.min(X) < 0 else np.log(x + 1))
        
        if self.seasonality_lag > 0:
            
            trend = X \
                    .rolling(self.seasonality_lag) \
                    .median() \
                    .values[self.seasonality_lag: ]

            pivot = X \
                    .reset_index()[self.seasonality_lag: ] \
                    .rename(columns = {0: 'y'})
                    
            pivot['detrended'] = pivot['y'] - trend
            pivot['period'] = pivot['index'] // self.seasonality_lag
            pivot['time'] = pivot['index'] % self.seasonality_lag

            seasonal = pivot[['period', 'time', 'detrended']] \
                            .set_index(['period', 'time']) \
                            .detrended.unstack(level = 0) \
                            .median(axis = 1)

            random = (pivot \
                        .set_index(['period', 'time'])['detrended'] - seasonal) \
                        .sort_index(level = [0, 1]) \
                        .values
        else:
            
            random = X.values[3:] - X.rolling(3) \
                                    .median() \
                                    .values[3:]
        
        self.bound = self._MAD(random)
        
        addition = self.seasonality_lag if self.seasonality_lag > 0 else 3
        
        self.peak_labels = np.argwhere(random > self.bound * self.alpha) \
                                    .reshape(-1) + addition
        self.collapse_labels = np.argwhere(random < self.bound * self.alpha * -1) \
                                    .reshape(-1) + addition

        
    def predict(self, anomaly_kind = 'all'):
        """
        Output the indices of original array those values are considered to be anomaly.
        Can output only peaks (values outreaching upper anomaly-bound) 
            either only collapses (values outreaching lower anomaly-bound)
        """
        
        if anomaly_kind == 'peak':
            return self.peak_labels
        elif anomaly_kind == 'collapse':
            return self.collapse_labels
        else:
            return np.sort(np.concatenate((self.peak_labels, self.collapse_labels)))