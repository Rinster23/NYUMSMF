def Garch11Fit(initparams,InputData):
    import scipy.optimize as scpo
    import numpy as np
    #Fit a GARCH(1,1) model to InputData using (9.43)
    #Returns the triplet a,b,c (actually a1, b1, c) from (9.42)
    #Initial guess is the triple in initparams

    array_data=np.array(InputData)

    def GarchMaxLike(params):
        import numpy as np        
        #Implement maximum likelihood formula Chapter 9
        xa,xb,xc=params
        if xa>10: xa=10
        if xb>10: xb=10
        if xc>10: xc=10
        #Use trick to force a and b between 0 and .999;
        #(a+b) less than .999; and c>0
        a=.999*np.exp(xa)/(1+np.exp(xa))
        b=(.999-a)*np.exp(xb)/(1+np.exp(xb))
        c=np.exp(xc)
        t=len(array_data)
        minimal=10**(-20)
        vargarch=np.zeros(t)

        #CHEATS!
        #Seed the variance with the whole-period variance
        #In practice we would have to have a holdout sample
        #at the beginning and roll the estimate forward.
        vargarch[0]=np.var(array_data)

        #Another cheat: take the mean over the whole period
        #and center the series on that. Hopefully the mean
        #is close to zero. Again in practice to avoid lookahead
        #we would have to roll the mean forward, using only
        #past data.
        overallmean=np.mean(array_data)
        #Compute GARCH(1,1) var's from data given parameters
        for i in range(1,t):
            #Note offset - i-1 observation of data
            #is used for i estimate of variance
            vargarch[i]=c+b*vargarch[i-1]+\
            a*(array_data[i-1]-overallmean)**2
            if vargarch[i]<=0:
                vargarch[i]=minimal
                
        #sum logs of variances
        logsum=np.sum(np.log(vargarch))
        #sum yi^2/sigma^2
        othersum=0
        for i in range(t):
            othersum+=((array_data[i]-overallmean)**2)/vargarch[i]
        #Actually -2 times objective since we are minimizing
        return(logsum+othersum)
    #End of GarchMaxLike

    #Transform parameters to the form used in GarchMaxLike
    #This ensures parameters are in bounds 0<a,b<1, 0<c
    aparam=np.log(initparams[0]/(.999-initparams[0]))
    bparam=np.log(initparams[1]/(.999-initparams[0]-initparams[1]))
    cparam=np.log(initparams[2])
    xinit=[aparam,bparam,cparam]
    #Run the minimization. Constraints are built-in.
    results = scpo.minimize(GarchMaxLike,
                            xinit,
                            method='CG')
    aparam,bparam,cparam=results.x
    a=.999*np.exp(aparam)/(1+np.exp(aparam))
    b=(.999-a)*np.exp(bparam)/(1+np.exp(bparam))
    c=np.exp(cparam)

    return([a,b,c])
#Done with Garch11Fit function