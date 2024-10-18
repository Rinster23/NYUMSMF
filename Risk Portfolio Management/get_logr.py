## get_logr.py

import qrpm_funcs as qf
import numpy as np

lastday=qf.LastYearEnd()
#Swiss franc, pound sterling, Japanese Yen
seriesnames=['DEXSZUS','DEXUSUK','DEXJPUS']
cdates,ratematrix=qf.GetFREDMatrix(seriesnames,enddate=lastday)
multipliers=[-1,1,-1]

lgdates,difflgs=qf.levels_to_log_returns(cdates,ratematrix,multipliers)

#Mean vector and covariance matrix are inputs to efficient frontier calculations
d=np.array(difflgs)
m=np.mean(d,axis=0)
c=np.cov(d.T)

#**************************************************************
chol=np.linalg.cholesky(c)
count=len(d)

#Generate random draws; use fixed seed to be replicable
from numpy.random import default_rng
rng = default_rng(12345678)

s_trial=rng.normal(0,1,size=[count,3])
logr_trial=np.matmul(chol,s_trial.T).T+m