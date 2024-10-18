#Code Segment 10.3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Get regional stock market index data from Ken French's website.
#Convert daily to Wednesday-Wednesday weekly.

ff_head='http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
ff_foot="_3_Factors_Daily_CSV.zip"
ff_names=["Europe","North_America","Japan"]

for name_index in range(len(ff_names)):
    print("Inputting ",ff_names[name_index])
    ffurl=ff_head+ff_names[name_index]+ff_foot
    #Skip the header rows
    df_region = pd.read_csv(ffurl, skiprows=6)
    #Standardize name of Date column and market return column
    col0=df_region.columns[0]
    df_region.rename(columns={col0:'Date'},inplace=True)
    df_region.rename(columns={"Mkt-RF":ff_names[name_index]},inplace=True)
    #Merge into aggregate
    if name_index == 0:
        df_returns=df_region[df_region.columns[0:2]]
    else:
        df_returns = df_returns.merge(df_region[df_region.columns[0:2]], 
                            left_on='Date', right_on='Date')

#Convert to log-returns
df_logs_day=np.log(1+df_returns[df_returns.columns[1:]]/100)

#Convert dates to datetime format
df_logs_day.insert(0,"Date",df_returns["Date"],True)
df_logs_day["Date"] = pd.to_datetime(df_logs_day["Date"], format='%Y%m%d')
        
#Convert log-returns to weekly (Wednesday-Wednesday)
#to avoid asynchronous trading effects
df_logs_day = df_logs_day.set_index("Date")
df_logs=df_logs_day.resample('W-Wed').sum()
#(Will include some holidays like July 4 and December 25, so a little off)

##Remove the partial year at the end
#lastyear=df_logs.index[-1].year
#df_logs.drop(df_logs.index[df_logs.index.year==lastyear],axis=0,inplace=True)

#Force agreement with book
lastyear=2021
df_logs.drop(df_logs.index[df_logs.index.year>lastyear],axis=0,inplace=True)

periodicity=52   #For use in later code segments

nobs=len(df_logs)
print(nobs," weekly observations starting",df_logs.index[0].strftime("%Y-%m-%d"), \
      "ending",df_logs.index[-1].strftime("%Y-%m-%d"))

#Code Segment 10.13
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
    plt.plot(range(len(stdgarch)),stdgarch,label=ticker)
    
plt.grid()
plt.title('Figure 10.9: GARCH(1,1) annualized standard deviations '+ \
          min(df_logs.index.strftime("%Y%m"))+'-'+ \
          str(max(df_logs.index.strftime("%Y%m"))))
plt.ylabel('GARCH SDs')
plt.legend()
stride=5*periodicity
tix=[x.strftime("%Y-%m-%d") for x in df_logs.index[0:len(df_logs)-1:stride]]
plt.xticks(range(0,len(df_logs),stride),tix,rotation=45)
plt.show()

for it,ticker in enumerate(tickerlist):
    print(ticker,'a=%1.4f' % gparams[it][0], \
               'b=%1.4f' % gparams[it][1], \
               'c=%1.8f' % gparams[it][2], \
               'AnnEquilibStd=%1.4f' % \
               np.sqrt(periodicity*gparams[it][2]/ \
                       (1-gparams[it][0]-gparams[it][1])))
#Code Segment 10.14
#Display before and after deGARCHing statistics

#Demeaned, DeGARCHed series go in dfeps
dfeps=df_logs.sort_values(by="Date").copy()
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
#Code Segment 10.6 (partial)

def plot_corrs(dates,corrs,corr_matrix,sccol,stride,title_str):
    #dates and corrs have same length
    #dates in datetime format
    #corrs is a list of correlation matrices
    #corr_matrix has the target correlations
    #names of securities are the column names of corr_matrix
    #sccol is colors for lines
    #stride is how many dates to skip between ticks on x-axis
    #title_str is title string

    nobs=len(corrs)
    nsecs=len(corrs[0])

    #plot correlations in corrs, nsec per time period
    ncorrs=nsecs*(nsecs-1)/2
    z=0
    #Go through each pair
    for j in range(nsecs-1):
        for k in range(j+1,nsecs):
            #form time series of sample correlations
            #for this pair of securities
            cs=[corrs[i][j,k] for i in range(nobs)]
            plt.plot(range(nobs),cs, \
                     label=corr_matrix.columns[j]+'/'+ \
                     corr_matrix.columns[k], \
                     color=sccol[z])
            #Show target correlation in same color
            line=[corr_matrix.iloc[j,k]]*(nobs)
            plt.plot(range(nobs),line,color=sccol[z])
            z+=1

    plt.legend()
    tix=[x.strftime("%Y-%m-%d") for x in dates[0:nobs+1:stride]]
    plt.xticks(range(0,nobs+1,stride),tix,rotation=45)
    plt.ylabel("Correlation")
    plt.title(title_str)
    plt.grid()
    plt.show();
#Done with plot_corrs
