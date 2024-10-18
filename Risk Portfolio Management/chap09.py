import pandas as pd
import numpy as np
import datetime
import qrpm_funcs as qf
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi']= 300

#Get all SPX options on CBOE as of December 31 of last year
lastday=qf.LastYearEnd()
df_opts=pd.read_excel(r"SPX_UnderlyingOptionsEODCalcs_"+lastday+".xlsx", \
                     engine="openpyxl")

#Subset the S&P 500 options with underlying SPX (eliminate SPXW, weekly expirations)
df_spx = df_opts[(df_opts.underlying_symbol == "^SPX") & (df_opts.root == "SPX")]

#Get S&P 500 price and quote date
spx_price = df_spx.active_underlying_price_1545.unique()[0]
quote_date = df_spx.quote_date.unique()[0]
stqd = str(quote_date)[:10]    #Display version YYYY-MM-DD

#Look between 80% of the money and 120% of the money
df_spx=df_spx[(df_spx.strike > .8*spx_price) & (df_spx.strike < 1.2*spx_price)]

#Eliminate expirations less than a week
df_spx=df_spx[df_spx.expiration>quote_date+np.timedelta64(6,'D')]
