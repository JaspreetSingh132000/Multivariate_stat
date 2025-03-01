#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from arch.unitroot.cointegration import DynamicOLS
from arch.unitroot.cointegration import FullyModifiedOLS

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

def OLS(x, y):
    est = sm.OLS(y.astype(float), x.astype(float)).fit()
    return est

def difference(vY):
    vYt = vY.diff().dropna()
    vYt.index = np.arange(0, len(vYt))
    return vYt

def main():
    df_dict = read_data()
    df_int = pd.DataFrame()

    for key in list(df_dict.keys()):
        df = df_dict[key]
        df_int[key] = df.iloc[:, 0]

    combinations = [
        ["Annual average of monthly total rainfall", "Annual average temperature"],
        ["Annual average of monthly total rainfall", "Agricultural GDP"],
        ["Annual average temperature", "Agricultural GDP"],
    ]

    print("MULTICOINTEGRATION")
    for i in combinations:
        print(i)
        mX = df_int[i]
        vec_rank_trace = select_coint_rank(mX, det_order=-1, k_ar_diff=1, method="trace", signif=0.05)
        print(vec_rank_trace)
        vec_rank_max = select_coint_rank(mX, det_order=-1, k_ar_diff=1, method="maxeig", signif=0.05)
        print(vec_rank_max)
        print("\n") 

    print("DOLS, FMOLS, ECM")
    for i in combinations:
        x = df_int[i[0]]
        y = df_int[i[1]]
        est_DOLS = DynamicOLS(x, y, trend="c").fit()
        print(i)
        print("DYNAMIC OLS:")
        print(f"parameter estimate: {est_DOLS.params[0]:.3f} with p-value: {est_DOLS.pvalues[0]:.3f}")
        
        est_FMOLS = FullyModifiedOLS(x, y, trend="c").fit()
        print("FULLY MODIFIED OLS: ")
        print(f"parameter estimate: {est_FMOLS.params[0]:.3f} with p-value: {est_FMOLS.pvalues[0]:.3f}")
        
        x_diff = difference(x)
        y_diff = difference(y)
        x_diff = sm.add_constant(x_diff)
        est_ECM = OLS(x_diff, y_diff)
        print("ECM:")
        print(f"parameter estimates: {est_ECM.params[1]:.3f} with p-value: {est_ECM.pvalues[1]:.3f}")
        print("\n")

if __name__ == "__main__":
    main()


# In[ ]:




