
*robustad* is a fully automated, flexible and algorithmically reliable Python module for time series anomaly detection.
This package is designed in scikit-learn style and provides minimum set of parameters to cope with. 
The original toolbox is considered to obtain robust and precise outlier detections inside time series data in the fast and "fit-predict" way

Table of Contents
-----------------

-  `Features`_
-  `Requirements`_
-  `Usage`_
-  `How does it work`_

Features
--------

-  *scikit-learn* style module
-  Can work in fully automated mode
-  Few hyperparameters easy to understand and tune
-  Comprehensive abilities to make precise predictions based on any type of time series
-  Common robustness, high level of interpretation 

Requirements
------------

In the latest version robustad requires *pandas* and *statsmodels*, so the one needs to have this modules be up to date,
consequently *numpy* and *scipy*

::

    pandas >= 0.24.2  
    statsmodels >= 0.10.1
    numpy >= 1.17.4
    scipy >= 1.3.2
  


Usage
-----

**Configuring the Model**

The robustad provides reasonable default parameters for making confident anomalies predictions within the usual meaning of outlier.
However the sensitivity of detection can be decreased or increased by parameter *alpha* tuning. Also if there is a clear evidence of seasonality or multiplicativity in given data, these conditions could be set explicitly. 

.. code:: python

    from robustad import RobustAnomalyDetector

    # Use default automatic detection
    model = RobustAnomalyDetector()
    # or custom detection
    model = RobustAnomalyDetector(alpha = 8, seasonality_lag = 7, log_transform = True)
 
**Training the Model**

After the model is configured the source data can be fit.

.. code:: python

    import numpy as np
    model.fit(np.array([i * 100 if (i % 7 == 0) else i for i in range(100)]))

**Making Predictions**

Now that the data is fit, we can identify indices (from 0 to len(array)) in the fitted data that contain anomalies.

.. code:: python

    >>> model.predict()
    array([0, 5, 10])
    >>> model.predict(anomaly_kind = 'peak')
    # Output only peaks (values that outreach upper bound of anomaly)
    array([0, 10])
    >>> model.predict(anomaly_kind = 'collapse')
    # Output only collapses (values that outreach lower bound of anomaly)
    array([5])

How does it work
---------

The main concept behind robustad is *median decomposition* of time series.
Once the algorithm gets a time series, it tries to decompose one into trend, seasonal and random part (close to STL-mean-decomposition does).
When random part (which is forced to be white noise like) is identified, median absolute deviation is calculated and bounds of anomaly values are obtained from multiplying MAD on hyperparameter alpha.

The whole algorithm can be described as sequence of the steps:

1) Trying to identify the optimal seasonality of time series using ACF (if seasonality parameter is unknown)
2) Removing trend and seasonaly (if there are) from original series
3) Identifying upper and lower bounds of anomaly values
4) Highlighting values that exceed any anomaly-bound


.. _Features: #features
.. _Requirements: #requirements
.. _Usage: #usage
.. _How does it work: #how-does-it-work
