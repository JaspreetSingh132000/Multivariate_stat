#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller,kpss
import arch.unitroot as pp
import warnings
warnings.filterwarnings('ignore')


def read_data():
    df = pd.read_excel("Downloads/MVE_Assignment_Dataset.xlsx")
    titles = ["Annual average of monthly total rainfall", "Annual average temperature", "Agricultural GDP"]
    columns = [5, 7, 9]
    df_dict = {}

    for title, col in zip(titles, columns):
        vY = pd.to_numeric(df.iloc[1:, col])
        log_vY = np.log(vY)
        df_dict[title] = pd.DataFrame({title: log_vY})

    return df_dict


def difference_series(vYt, vY):
    vYt = vY.diff().dropna()
    vYt.index = np.arange(0, len(vYt))
    return vYt

def no_trend_test(vYt,ytmin1):
    "No trend"
    est = OLS(ytmin1,vYt)
    dEst = est.params[0]
    dStd = est.bse[0]
    t = dEst/dStd
    residuals = est.resid
    durbin = durbin_watson(residuals,axis=0)
    
    return t,durbin,residuals, "no trend no constant"
    
def constant_test(vYt, ytmin1):
    ytmin1 = sm.add_constant(ytmin1)
    est = OLS(ytmin1, vYt)
    dEst = est.params[1]
    dStd = est.bse[1]
    t = dEst / dStd
    residuals = est.resid
    durbin = durbin_watson(residuals, axis=0)
    return t, durbin, residuals, "constant, no trend"

def constant_and_trend_test(vYt, ytmin1):
    ytmin1 = sm.add_constant(ytmin1)
    ytmin1["trend"] = np.arange(0, len(ytmin1))
    est = OLS(ytmin1, vYt)
    dEst = est.params[1]
    dStd = est.bse[1]
    t = dEst / dStd
    residuals = est.resid
    durbin = durbin_watson(residuals, axis=0)
    
    return t, durbin, residuals, "constant and trend"
    
def OLS(x, y):
    est = sm.OLS(y.astype(float),x.astype(float)).fit()

    return est


def check_rejection(t, crit_val):
    if t<crit_val:
        return "rejected"
    else:
        return "not rejected"
    
def check_correlation(durbin):
    if np.abs(durbin-2)<0.5:
        print("NO CORRELATION IN ERRORS ")
        return "no correlation"
    else:
        print("CORRELATION IN ERRORS ")
        return "correlation"
    
def check_ADF(vY, regression):
    ADF = adfuller(vY, maxlag=3, regression=regression)
    if ADF[0]<ADF[4]["5%"]:
        print(f"rejected ADF test statistic: {ADF[0]:.3f}, crit value: {ADF[4]['5%']:.3f}")

        return "rejected"
    else:
        print(f"not rejected ADF test statistic: {ADF[0]:.3f}, crit value: {ADF[4]['5%']:.3f}")

        return "not rejected"
    
def check_KPSS(vY, regression):
    kpss_result = kpss(vY,nlags=3,regression=regression)
    if kpss_result[0]<kpss_result[3]["5%"]:
        print(f"rejected KPSS test statistic: {kpss_result[0]:.3f}, crit value: {kpss_result[3]['5%']:.3f}")
        return "rejected"
    else:
        print(f"not rejected KPSS test statistic: {kpss_result[0]:.3f}, crit value: {kpss_result[3]['5%']:.3f}")

        return "not rejected"
    
def check_Phillips(vY, regression):
    philips_result = pp.PhillipsPerron(vY,trend = regression)
    if philips_result.pvalue <0.05:
        print(f"rejected Phillips Perron p-value: {philips_result.pvalue:.3f}, significance: 0.05")
        return "rejected"
    else:
        print(f"not rejected Phillips Perron p-value: {philips_result.pvalue:.3f}, significance: 0.05")
        return "not rejected"
    
    
def result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, currentRegression, regression, i, j):
    vRegression = ["c", "ct"]
    vRegression2 = [constant_test, constant_and_trend_test] 
    crit_value = -2.93

    i_DF = check_rejection(t, crit_value)
    print(f"I({j})", DF)
    print(f"{i_DF} Dickey-Fuller test statistic: {t:.3f} crit value: {crit_value:.3f}")

    if i_DF == "rejected":
        i_corr = check_correlation(durbin)
        if i_corr == "no correlation":
            print(f"Rejected null of {column} {DF} I({j}) By normal Dickey-Fuller and no autocorrelation in residuals.")
        elif i_corr == "correlation":
            i_ADF = check_ADF(vY, regression)
            if i_ADF == "rejected":
                print(f"Rejected null of {column} {DF} I({j}) By augmented Dickey-Fuller with autocorrelation in residuals.")
            elif i_ADF == "not rejected" and i <= 1:
                i_kpss = check_KPSS(vY, "c" if regression == "nc" else "ct")
                if i_kpss == "not rejected":
                    t, durbin, residuals, DF = vRegression2[i](vYt, ytmin1)
                    i += 1
                    result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, vRegression2[i - 1], vRegression[i - 1], i, j)
                elif i_kpss == "rejected":
                    i_pp = check_Phillips(vY, "n" if regression == "nc" else regression)
                    if i_pp == "rejected":
                        t, durbin, residuals, DF = vRegression2[i](vYt, ytmin1)
                        i += 1
                        result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, vRegression2[i - 1], vRegression[i - 1], i, j)
                    elif i_pp == "not rejected":
                        print("Differenced")
                        j += 1
                        vY = vYt.copy()
                        ytmin1 = vY.iloc[0:-1]
                        vYt = difference_series(vYt, vYt)
                        t, durbin, residuals, DF = currentRegression(vYt, ytmin1)
                        result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, currentRegression, regression, i, j)

    elif i_DF == "not rejected":
        i_corr = check_correlation(durbin)
        if i_corr == "no correlation":
            print("Differenced")
            j += 1
            vY = vYt.copy()
            ytmin1 = vY.iloc[0:-1]
            vYt = difference_series(vYt, vYt)
            t, durbin, residuals, DF = currentRegression(vYt, ytmin1)
            result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, currentRegression, regression, i, j)
        elif i_corr == "correlation":
            i_ADF = check_ADF(vY, regression)
            if i_ADF == "rejected":
                print(f"Rejected null of {key} {column} {DF} I({j}) By augmented Dickey-Fuller and no autocorrelation in residuals.")
            elif i_ADF == "not rejected" and i <= 1:
                i_kpss = check_KPSS(vY, "c" if regression == "nc" else "ct")
                if i_kpss == "not rejected":
                    t, durbin, residuals, DF = vRegression2[i](vYt, ytmin1)
                    i += 1
                    result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, vRegression2[i - 1], vRegression[i - 1], i, j)
                elif i_kpss == "rejected":
                    i_pp = check_Phillips(vY, "n" if regression == "nc" else regression)
                    if i_pp == "rejected":
                        t, durbin, residuals, DF = vRegression2[i](vYt, ytmin1)
                        i += 1
                        result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, vRegression2[i - 1], vRegression[i - 1], i, j)
                    elif i_pp == "not rejected":
                        print("Differenced")
                        j += 1
                        vY = vYt.copy()
                        ytmin1 = vY.iloc[0:-1]
                        vYt = difference_series(vYt, vYt)
                        t, durbin, residuals, DF = currentRegression(vYt, ytmin1)
                        result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, currentRegression, regression, i, j)

                    
def main():
    df_dict = read_data()

    
    for key in list(df_dict.keys()):
        df = df_dict[key]
        print(key + ":")
        
        for column in df.columns.values:
            regression = "nc"
            vY = df[column]
            vY.index = np.arange(0, len(vY))
            vYt = difference_series(vY, vY)
            ytmin1 = vY.iloc[0:-1]
            currentRegression = constant_and_trend_test
            t, durbin, residuals, DF = currentRegression(vYt, ytmin1)
            result(t, durbin, residuals, key, column, DF, vY, vYt, ytmin1, currentRegression, regression, 0, 0)
            print("\n")
                
                
        
        
if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




