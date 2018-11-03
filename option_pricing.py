import numpy as np
from math import *
from scipy import stats
import math



# ========================Task 1:European Option===========================
def price_call(S, K, r, q, sigma, tau):  # calculate the call option price according to formula
    d1 = (log(S / K) + (r - q) * tau) / (sigma * sqrt(tau)) + sigma * sqrt(tau) / 2.0
    d2 = (log(S / K) + (r - q) * tau) / (sigma * sqrt(tau)) - sigma * sqrt(tau) / 2.0
    return S * exp(-q * tau) * stats.norm.cdf(d1) - K * exp(-r * tau) * stats.norm.cdf(d2)


def price_put(S, K, r, q, sigma, tau):  # calculate the put option price according to formula
    d1 = (log(S / K) + (r - q) * tau) / (sigma * sqrt(tau)) + sigma * sqrt(tau) / 2.0
    d2 = (log(S / K) + (r - q) * tau) / (sigma * sqrt(tau)) - sigma * sqrt(tau) / 2.0
    return K * exp(-r * tau) * stats.norm.cdf(-d2) - S * exp(-q * tau) * stats.norm.cdf(-d1)


# =====================Task 1 End==========================

# ========================Task 2:Implied Vol===========================

def vega(S, K, r, q, sigma, tau):
    d1 = (log(S / K) + (r - q) * tau) / (sigma * sqrt(tau)) + sigma * sqrt(tau) / 2.0
    return S * exp(-q * tau) * sqrt(tau) * stats.norm.pdf(d1)


def v_call(S, r, q, tau, K, C_true):  # calculate the volatility of call option

    # check upper bounds and lower bounds
    if C_true < max(S * exp(-q * tau) - K * exp(-r * tau), 0) or C_true > S * exp(-q * tau):
        return np.nan

    sigma = sqrt(2.0 * abs((log(S / K) + (r - q) * tau) / tau))  # starting value
    tol = 1e-8
    sigmadiff = 1.0
    n = 1
    nmax = 100

    while sigmadiff >= tol and n < nmax:
        C = price_call(S, K, r, q, sigma, tau)
        Cvega = vega(S, K, r, q, sigma, tau)
        if Cvega < tol:
            return np.nan
        increment = (C - C_true) / Cvega
        sigma = sigma - increment
        n = n + 1
        sigmadiff = abs(increment)

    if n >= nmax:  # not converge
        return np.nan
    return sigma


def v_put(S, r, q, tau, K, P_true):  # calculate the volatility of put option

    # check upper bounds and lower bounds
    if P_true < max(K * exp(-r * tau) - S * exp(-q * tau), 0) or P_true > K * exp(-r * tau):
        return np.nan

    sigma = sqrt(2.0 * abs((log(S / K) + (r - q) * tau) / tau))  # starting value
    tol = 1e-8
    sigmadiff = 1.0
    n = 1
    nmax = 100

    while sigmadiff >= tol and n < nmax:
        P = price_put(S, K, r, q, sigma, tau)
        Pvega = vega(S, K, r, q, sigma, tau)
        increment = (P - P_true) / Pvega
        if Pvega < tol:
            return np.nan
        sigma = sigma - increment
        n = n + 1
        sigmadiff = abs(increment)
    if n >= nmax:  # not converge
        return np.nan
    return sigma


# =====================Task 2 End==========================

# ========================Task 3:Geometric===========================

def geometric_Asian_call_option(S, sigma, r, T, K, n):
    sigma_g = sigma * sqrt(1.0 * (n + 1) * (2 * n + 1) / (6 * n * n))
    miu_g = (r - 1.0 / 2 * sigma * sigma) * (n + 1) / (2 * n) + 1.0 / 2 * sigma_g * sigma_g
    d1_g = (log(S / K) + (miu_g + 1.0 / 2 * sigma_g * sigma_g) * T) / (sigma_g * sqrt(T))
    d2_g = d1_g - sigma_g * sqrt(T)
    option = exp(-r * T) * (S * exp(miu_g * T) * stats.norm.cdf(d1_g) - K * stats.norm.cdf(d2_g))
    return option


def geometric_Asian_put_option(S, sigma, r, T, K, n):
    sigma_g = sigma * sqrt(1.0 * (n + 1) * (2 * n + 1) / (6 * n * n))
    miu_g = (r - 1.0 / 2 * sigma * sigma) * (n + 1) / (2 * n) + 1.0 / 2 * sigma_g * sigma_g
    d1_g = (log(S / K) + (miu_g + 1.0 / 2 * sigma_g * sigma_g) * T) / (sigma_g * sqrt(T))
    d2_g = d1_g - sigma_g * sqrt(T)
    option = exp(-r * T) * (K * stats.norm.cdf(-d2_g) - S * exp(miu_g * T) * stats.norm.cdf(-d1_g))
    return option


def geometric_basket_call_option(S1, S2, sigma1, sigma2, r, T, K, rou):
    sigma_b = sqrt(sigma1 * sigma1 + 2 * sigma1 * sigma2 * rou + sigma2 * sigma2) / 2
    miu_b = r - 1.0 / 2 * (sigma1 * sigma1 + sigma2 * sigma2) / 2 + 1.0 / 2 * sigma_b * sigma_b
    basket = sqrt(S1 * S2)
    d1_b = (log(basket / K) + (miu_b + 1.0 / 2 * sigma_b * sigma_b) * T) / (sigma_b * sqrt(T))
    d2_b = d1_b - sigma_b * sqrt(T)
    option = exp(-r * T) * (basket * exp(miu_b * T) * stats.norm.cdf(d1_b) - K * stats.norm.cdf(d2_b))
    return option


def geometric_basket_put_option(S1, S2, sigma1, sigma2, r, T, K, rou):
    sigma_b = sqrt(sigma1 * sigma1 + 2 * sigma1 * sigma2 * rou + sigma2 * sigma2) / 2
    miu_b = r - 1.0 / 2 * (sigma1 * sigma1 + sigma2 * sigma2) / 2 + 1.0 / 2 * sigma_b * sigma_b
    basket = sqrt(S1 * S2)
    d1_b = (log(basket / K) + (miu_b + 1.0 / 2 * sigma_b * sigma_b) * T) / (sigma_b * sqrt(T))
    d2_b = d1_b - sigma_b * sqrt(T)
    option = exp(-r * T) * (K * stats.norm.cdf(-d2_b) - basket * exp(miu_b * T) * stats.norm.cdf(-d1_b))
    return option


def geometric_Asian_option(S, sigma, r, T, K, n, option_type):
    if option_type == "C" or option_type == "CALL":
        return geometric_Asian_call_option(S, sigma, r, T, K, n)
    else:
        return geometric_Asian_put_option(S, sigma, r, T, K, n)


def geometric_basket_option(S1, S2, sigma1, sigma2, r, T, K, rou, option_type):
    if option_type == "C" or option_type == "CALL":
        return geometric_basket_call_option(S1, S2, sigma1, sigma2, r, T, K, rou)
    else:
        return geometric_basket_put_option(S1, S2, sigma1, sigma2, r, T, K, rou)

# ============Task 4:Arithmetic Asian option================
def ArithmeticAsianOptionMC(T, K, n, S, r, sigma, option_type, NumPath, cv):

    np.random.seed(0)
    SPath = np.zeros((NumPath, n), dtype = float)
    randn = np.zeros((NumPath, n), dtype = float)
    Dt = T / n
    drift = exp((r - 0.5 * sigma **2)*Dt)
    c2 = sigma * np.sqrt(Dt)

    for i in range(NumPath):
        randn[i, :] = np.random.standard_normal(n)

    growthFactor=drift * np.exp(sigma * np.sqrt(Dt) * randn[:,0])
    SPath[:, 0] = S *   growthFactor                      #initialize first step result

    for i in range(1, n):                                 #simulate the remaining steps in monte carlo
        S_1 = SPath[:, i - 1]
        growthFactor = drift * np.exp(sigma * np.sqrt(Dt) * randn[:, i])
        SPath[:, i] = S_1* growthFactor

    # Arithmetic mean
    arithMean = SPath.mean(1)
    geoMean = np.exp(1 / n * np.log(SPath).sum(1))

    if option_type == "C"or option_type == "CALL":
        arithPayoff = np.exp(-r*T)*np.maximum(arithMean - K, 0)   #payoffs
        geoPayoff = np.exp(-r*T)*np.maximum(geoMean - K, 0)
    else:
        arithPayoff = np.maximum(K - arithMean, 0)*np.exp(-r*T)
        geoPayoff = np.maximum(K - geoMean, 0)*np.exp(-r*T)

    #Standard Monte Carlo
    if cv == "NO":
        Pmean=np.mean(arithPayoff)
        Pstd=np.std(arithPayoff)
        confmc=[Pmean-1.96*Pstd/np.sqrt(NumPath),Pmean+1.96*Pstd/np.sqrt(NumPath)]
        OptionValue=Pmean
        return OptionValue, confmc

    #Control variates
    else:
        covXY = np.mean(arithPayoff*geoPayoff) - (np.mean(geoPayoff) * np.mean(geoPayoff))
        theta = covXY/np.var(geoPayoff)
        geo = geometric_Asian_option(S, sigma, r, T, K, n, option_type)
        Z = arithPayoff + theta*(geo - geoPayoff)
        Zmean=np.mean(Z)
        Zstd=np.std(Z)
        OptionValue=Zmean
        Confcv=[Zmean-1.96*Zstd/np.sqrt(NumPath),Zmean+1.96*Zstd/np.sqrt(NumPath)]
        return OptionValue,Confcv


#Arithmetic basket option
def ArithmeticBasketOptionMC(T, K, S1, S2, r, sigma1, sigma2, rou, option_type, NumPath, cv):
    np.random.seed(0)
    z_1 = np.random.standard_normal(NumPath)
    z = np.random.standard_normal(NumPath)
    z_2 = rou * z_1 + np.sqrt(1 - rou ** 2) * z
    S1_T = S1 * np.exp((r - 0.5 * sigma1 ** 2) * T + sigma1 * np.sqrt(T) * z_1)
    S2_T = S2 * np.exp((r - 0.5 * sigma2 ** 2) * T + sigma2 * np.sqrt(T) * z_2)
    S_T = (S1_T + S2_T) / 2
    S_S = np.exp((np.log(S1_T) + np.log(S2_T)) / 2)
    if  option_type == "C"or option_type == "CALL":
        arithPayoff = np.exp(-r * T)*(((S1_T + S2_T) / 2) - K)
        geoPayoff = np.exp(-r * T)*((np.exp((np.log(S1_T) + np.log(S2_T)) / 2)) - K)
    else:
        arithPayoff = np.exp(-r * T) * (K - S_T)
        geoPayoff = np.exp(-r * T) * (K - S_S)

    for i in range(0, NumPath):
        arithPayoff[i] = max(arithPayoff[i], 0)
        geoPayoff[i] = max(geoPayoff[i], 0)

    # Standard Monte Carlo
    if cv == 'NO':
        Pmean = np.mean(arithPayoff)
        Pstd = np.std(arithPayoff)
        Confmc = [Pmean - 1.96 * Pstd / np.sqrt(NumPath), Pmean + 1.96 * Pstd / np.sqrt(NumPath)]
        OptionValue = Pmean
        return round(OptionValue, 5), Confmc

    # Monte Carlo with control variates
    else:
        X_Y = [0.0] * NumPath
        for i in range(0, NumPath):
            X_Y[i] = arithPayoff[i] * geoPayoff[i]
        covXY = np.mean(X_Y) - np.mean(arithPayoff) * np.mean(geoPayoff)
        theta = covXY / np.var(geoPayoff)

        geo = geometric_basket_option(S1, S2, sigma1, sigma2, r, T, K, rou, option_type)
        Z = arithPayoff + theta * (geo - geoPayoff)
        Zmean = np.mean(Z)
        Zstd = np.std(Z)
        OptionValue = Zmean
        Confcv = [Zmean - 1.96 * Zstd / np.sqrt(NumPath), Zmean + 1.96 * Zstd / np.sqrt(NumPath)]
        return round(OptionValue, 5), Confcv

# ====================Task 5 :Amercian Option==============================

def Bionomial_tree(S,K,r,sigma,T,N,type):
    delta = T/N
    u = math.exp(sigma * math.sqrt(delta))
    d = 1/u
    p = (math.exp(r*delta)-d)/(u - d)

    s = np.zeros(N + 1)  # Stock Price: Array
    value = np.zeros(N + 1)  # Option value: Array

    # For example N = 2
    for i in range(N+1):
        s[i] = S * u**(N-i) * d**(i)
        # Time to maturity:
        # s[0] = S * u * u
        # s[1] = S * u * d
        # s[2] = S * d * d
    if type =='C':
        for vi in range(N+1):
            value[vi] = max(s[vi] - K,0)
        # Time to maturity:
        # s[0] - value[0]
        # s[1] - value[1]
        # s[2] - value[2]
        for vj in range(N):
            for k in range(N - vj):
                s[k] = s[k]/u  # st[0] = S * u   st[1] = S * d   st[0] = S
                value[k] = max(s[k] - K,math.exp(-r * delta) * (p * value[k] + (1-p) * value[k+1]))
        return value[0]
    if type =='P':
        for vi in range(N+1):
            value[vi] = max(K - s[vi],0)

        for vj in range(N):
            for k in range(N - vj):
                s[k] = s[k]/u  # st[0] = S * u   st[1] = S * d   st[0] = S
                value[k] = max(K - s[k],math.exp(-r * delta) * (p * value[k] + (1-p) * value[k+1]))
        return value[0]

# =====================Task 5 End==========================

if __name__=="__main__":
    print("=================European Option==============")
    print("S=100, K=100, r=1%, sigma=20%, T=0.5, q=5%, option_type=Call:")
    print(price_call(100, 100, 0.01, 0.05, 0.2, 0.5))
    print("S=100, K=100, r=1%, sigma=20%, T=0.5, q=3%, option_type=Call:")
    print(price_call(100, 100, 0.01, 0.03, 0.2, 0.5))
    print("S=100, K=100, r=1%, sigma=20%, T=0.5, q=5%, option_type=Put:")
    print(price_put(100, 100, 0.01, 0.05, 0.2, 0.5))
    print("S=100, K=100, r=1%, sigma=20%, T=0.5, q=3%, option_type=Put:")
    print(price_put(100, 100, 0.01, 0.03, 0.2, 0.5))

    print("=================Geometric Asian Option==============")
    print("S=100, K=100, r=5%, sigma=30%, T=3, n=50, option_type=Call:")
    print(geometric_Asian_call_option(100, 0.3, 0.05, 3, 100, 50))
    print("S=100, K=100, r=5%, sigma=30%, T=3, n=100, option_type=Call:")
    print(geometric_Asian_call_option(100, 0.3, 0.05, 3, 100, 100))
    print("S=100, K=100, r=5%, sigma=40%, T=3, n=50, option_type=Call:")
    print(geometric_Asian_call_option(100, 0.4, 0.05, 3, 100, 50))
    print("S=100, K=100, r=5%, sigma=30%, T=3, n=50, option_type=Put:")
    print(geometric_Asian_put_option(100, 0.3, 0.05, 3, 100, 50))
    print("S=100, K=100, r=5%, sigma=30%, T=3, n=100, option_type=Put:")
    print(geometric_Asian_put_option(100, 0.3, 0.05, 3, 100, 100))
    print("S=100, K=100, r=5%, sigma=40%, T=3, n=50, option_type=Put:")
    print(geometric_Asian_put_option(100, 0.4, 0.05, 3, 100, 50))

    print("=================Geometric Basket Option==============")
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=100,correaltion=0.5, option_type=Call")
    print(geometric_basket_call_option(100, 100, 0.3, 0.3, 0.05, 3, 100, 0.5))
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=100,correaltion=0.9, option_type=Call")
    print(geometric_basket_call_option(100, 100, 0.3, 0.3, 0.05, 3, 100, 0.9))
    print("S1=100,S2=100,sigma1=0.1,sigma2=0.3,r=0.05,T=3,K=100,correaltion=0.5, option_type=Call")
    print(geometric_basket_call_option(100, 100, 0.1, 0.3, 0.05, 3, 100, 0.5))
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=80,correaltion=0.5, option_type=Call")
    print(geometric_basket_call_option(100, 100, 0.3, 0.3, 0.05, 3, 80, 0.5))
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=120,correaltion=0.5, option_type=Call")
    print(geometric_basket_call_option(100, 100, 0.3, 0.3, 0.05, 3, 120, 0.5))
    print("S1=100,S2=100,sigma1=0.5,sigma2=0.5,r=0.05,T=3,K=100,correaltion=0.5, option_type=Call")
    print(geometric_basket_call_option(100, 100, 0.5, 0.5, 0.05, 3, 100, 0.5))

    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=100,correaltion=0.5, option_type=Put")
    print(geometric_basket_put_option(100, 100, 0.3, 0.3, 0.05, 3, 100, 0.5))
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=100,correaltion=0.9, option_type=Put")
    print(geometric_basket_put_option(100, 100, 0.3, 0.3, 0.05, 3, 100, 0.9))
    print("S1=100,S2=100,sigma1=0.1,sigma2=0.3,r=0.05,T=3,K=100,correaltion=0.5, option_type=Put")
    print(geometric_basket_put_option(100, 100, 0.1, 0.3, 0.05, 3, 100, 0.5))
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=80,correaltion=0.5, option_type=Put")
    print(geometric_basket_put_option(100, 100, 0.3, 0.3, 0.05, 3, 80, 0.5))
    print("S1=100,S2=100,sigma1=0.3,sigma2=0.3,r=0.05,T=3,K=120,correaltion=0.5, option_type=Put")
    print(geometric_basket_put_option(100, 100, 0.3, 0.3, 0.05, 3, 120, 0.5))
    print("S1=100,S2=100,sigma1=0.5,sigma2=0.5,r=0.05,T=3,K=100,correaltion=0.5, option_type=Put")
    print(geometric_basket_put_option(100, 100, 0.5, 0.5, 0.05, 3, 100, 0.5))

    r = 0.05
    T = 3
    S = 100
    print("===========================Asian option from Standard MC=====================")
    print("PUT")
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.3, 'PUT', 100000,'No'))
    print(ArithmeticAsianOptionMC(T,100,100,S, r, 0.3, 'PUT', 100000,'No'))
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.4, 'PUT', 100000,'No'))
    print("CALL")
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.3, 'CALL', 100000,'No'))
    print(ArithmeticAsianOptionMC(T,100,100,S, r, 0.3, 'CALL', 100000,'No'))
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.4, 'CALL', 100000,'No'))
    print("===========================Asian option from MC with Control =====================")
    print("PUT")
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.3, 'PUT', 100000,'YES'))
    print(ArithmeticAsianOptionMC(T,100,100,S, r, 0.3, 'PUT', 100000,'YES'))
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.4, 'PUT', 100000,'YES'))
    print("CALL")
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.3, 'CALL', 100000,'YES'))
    print(ArithmeticAsianOptionMC(T,100,100,S, r, 0.3, 'CALL', 100000,'YES'))
    print(ArithmeticAsianOptionMC(T,100,50,S, r, 0.4, 'CALL', 100000,'YES'))
    print("============================Arithmetic Basket from Standard MC=====================")
    print("PUT")
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.9, 'PUT', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.1, 0.3, 0.5, 'PUT', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 80, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 120, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.5, 0.5, 0.5, 'PUT', 100000,'No'))
    print("CALL")
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.5, 'CALL', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.9, 'CALL', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.1, 0.3, 0.5, 'CALL', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 80, 100, 100, r, 0.3, 0.3, 0.5, 'CALL', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 120, 100, 100, r, 0.3, 0.3, 0.5, 'CALL', 100000,'No'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.5, 0.5, 0.5, 'CALLl', 100000,'No'))
    print("============================Arithmetic Basket from MC with Control Variate=====================")
    print("PUT")
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.9, 'PUT', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.1, 0.3, 0.5, 'PUT', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 80, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 120, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.5, 0.5, 0.5, 'PUT', 100000, 'YES'))
    print("CALL")
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.5, 'CALL', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.9, 'CALL', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.1, 0.3, 0.5, 'CALL', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 80, 100, 100, r, 0.3, 0.3, 0.5, 'CALL', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 120, 100, 100, r, 0.3, 0.3, 0.5, 'CALL', 100000, 'YES'))
    print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.5, 0.5, 0.5, 'CALLl', 100000, 'YES'))

    print("=============================American option=============================")
    print("Step = 3")
    print("CALL")
    # Bionomial_tree(S, K, r, sigma, T, N, type)
    print(Bionomial_tree(50, 50, 0.05, 0.3, 0.25, 3, 'C'))
    print(Bionomial_tree(60, 50, 0.05, 0.3, 0.25, 3, 'C'))
    print(Bionomial_tree(50, 60, 0.05, 0.3, 0.25, 3, 'C'))
    print(Bionomial_tree(50, 50, 0.08, 0.3, 0.25, 3, 'C'))
    print(Bionomial_tree(50, 50, 0.05, 0.5, 0.25, 3, 'C'))
    print(Bionomial_tree(50, 50, 0.05, 0.3, 0.5, 3, 'C'))
    print("PUT")
    print(Bionomial_tree(50, 50, 0.05, 0.3, 0.25, 3, 'P'))
    print(Bionomial_tree(60, 50, 0.05, 0.3, 0.25, 3, 'P'))
    print(Bionomial_tree(50, 60, 0.05, 0.3, 0.25, 3, 'P'))
    print(Bionomial_tree(50, 50, 0.08, 0.3, 0.25, 3, 'P'))
    print(Bionomial_tree(50, 50, 0.05, 0.5, 0.25, 3, 'P'))
    print(Bionomial_tree(50, 50, 0.05, 0.3, 0.5, 3, 'P'))
    import os
    os.system("pause")

