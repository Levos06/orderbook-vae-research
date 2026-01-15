import numpy as np

def fit_trend_ols(prices):
    """
    Fits a linear trend using Ordinary Least Squares for each sample.
    prices: (N, 800)
    Returns: m (N, 1), b (N, 1) such that trend = m*x + b
    """
    N, L = prices.shape
    x = np.arange(L)
    # Solve for each row: y = Ap where A = [x 1]
    # We can do this vectorized for all rows
    A = np.vstack([x, np.ones(L)]).T # (L, 2)
    
    # Using np.linalg.lstsq in a loop or vectorized
    # Vectorized: p = (A^T A)^-1 A^T y
    ATA_inv_AT = np.linalg.inv(A.T @ A) @ A.T # (2, L)
    params = prices @ ATA_inv_AT.T # (N, 2)
    
    return params[:, 0:1], params[:, 1:2]

def fit_trend_endpoints(prices):
    """
    Fits a linear trend by connecting the first and last points.
    prices: (N, 800)
    Returns: m (N, 1), b (N, 1)
    """
    N, L = prices.shape
    y0 = prices[:, 0:1]
    yL = prices[:, -1:]
    
    m = (yL - y0) / (L - 1)
    b = y0
    
    return m, b

def get_trend(m, b, length=800):
    """
    Generates the trend line.
    m, b: (N, 1)
    """
    x = np.arange(length) # (L,)
    return m * x + b # (N, L)

def detrend_prices(prices, method='ols'):
    if method == 'ols':
        m, b = fit_trend_ols(prices)
    elif method == 'endpoints':
        m, b = fit_trend_endpoints(prices)
    else:
        raise ValueError("Method must be 'ols' or 'endpoints'")
    
    trend = get_trend(m, b, prices.shape[1])
    residuals = prices - trend
    return residuals, m, b
