#Code Segment 6.1 (section)
import numpy as np
import pandas as pd
import qrpm_funcs as qf

#Get 4 currencies until the end of previous year.
firstday='1999-01-04'
lastday=qf.LastYearEnd()
seriesnames=['DEXSZUS','DEXUSUK','DEXJPUS','DEXUSEU']
cdates,ratematrix=qf.GetFREDMatrix(seriesnames,
            startdate=firstday,enddate=lastday)
multipliers=[-1,1,-1,1]

lgdates,difflgs=qf.levels_to_log_returns(cdates,ratematrix,multipliers)

#Put data in df_logs so it looks like Chapter 10 data
from datetime import datetime

df_logs = pd.DataFrame(difflgs, \
            index=[datetime.strptime(lg,"%Y-%m-%d") for lg in lgdates], \
            columns=seriesnames)
periodicity=252   #For use in later code segments
nobs=len(df_logs)
print(nobs," daily observations starting",df_logs.index[0], \
      "ending",df_logs.index[-1])

#Code Segment 10.13 (truncated)
import scipy.stats as spst
from scipy.optimize import minimize_scalar

#CHEAT! - get overall mean and standard deviation vectors
#In practice, would need to do everything out of sample - 
#start with a learning sample, e.g.
overallmean=np.mean(df_logs,axis=0)
overallstd=np.std(df_logs)
tickerlist=df_logs.columns

#Get GARCH params for each ticker
initparams=[.12,.85,.6]
gparams=[qf.Garch11Fit(initparams,df_logs[ticker]) for ticker in tickerlist]

minimal=10**(-20)
stgs=[] #Save the running garch sigmas
for it,ticker in enumerate(tickerlist):
    a,b,c=gparams[it]
    
    #Create time series of sigmas
    t=len(df_logs[ticker])
    stdgarch=np.zeros(t)
    stdgarch[0]=overallstd[ticker]
    #Compute GARCH(1,1) stddev's from data given parameters
    for i in range(1,t):
        #Note offset - i-1 observation of data
        #is used for i estimate of std deviation
        previous=stdgarch[i-1]**2
        var=c+b*previous+\
            a*(df_logs[ticker][i-1]-overallmean[ticker])**2
        stdgarch[i]=np.sqrt(var)

    #Save for later de-GARCHing
    stgs.append(stdgarch)
    stdgarch=100*np.sqrt(periodicity)*stgs[it]  #Annualize

#Code Segment 10.14
#Display before and after deGARCHing statistics

#Demeaned, DeGARCHed series go in dfeps
dfeps=df_logs.sort_index().copy()
for it,ticker in enumerate(tickerlist):
    dfeps[ticker]-=overallmean[ticker]
    for i in range(len(dfeps)):
        dfeps[ticker].iloc[i]/=stgs[it][i]
    print(ticker)
    print('    DeGARCHed Mean:',np.mean(dfeps[ticker]))
    
    print('    Raw annualized Std Dev:',np.sqrt(periodicity)*overallstd[ticker])
    print('    DeGARCHed Std Dev:',np.std(dfeps[ticker]))
    
    print('    Raw excess kurtosis:',spst.kurtosis(df_logs[ticker]))
    print('    DeGARCHed Excess Kurtosis:',spst.kurtosis(dfeps[ticker]))