import numpy
import random
import pylab as plt
from pylab import log10, sqrt, pi, log,exp
from numpy import tanh, sinh, arctanh
from scipy.special import erf
import pickle
from scipy.interpolate import BPoly
from scipy.special import hyp2f1, gamma
from scipy.integrate import quad, simpson, ode
import sys

SMALLNUMBER=1e-9

class aEMS:
    def __init__(self, Sigma, MGC0, FEH, **kwargs):
        
        # 3 required parameters:
        self.Sigma = Sigma
        self.MGC0 = MGC0
        self.FEH = FEH
        
        self._set_kwargs(**kwargs)
        self._set_mf()
        self._initialise()
        self._evolve()
        self._clean_up()

    def _set_kwargs(self, **kwargs):
        """ Set parameters and scales, all these variables can be changed via the command line """
        self.seed = 123

        self.nt = 10  # Number of output times
        self.ms = 1   # [Msun] seed mass
        self.mw = 100 # [Msun] reference mass for mdw wind parameter

        self.mdwc = 30
        self.SFE = 0.1 # Hardcoded
        self.Nmc = 1e5 # Number of mass points

        self.mlim = 100  # minimum mass to be considered for wind
        self.tnuc = 3 # [Myr] nuclear time, fix for now

        self.a = -2.3 # mass function slope
        
        # Numerical 
        self.m0 = 10 # trial to mass for root finding
        self.nm = 200 # number of mass points
        self.do_psi = True

        self.do_clean = True
        
        self.initial_abundances=[0,0.4,0]
        self.verbose = False
        
        if kwargs is not None:
            for key, value in kwargs.items():
                # Check for scaling input
                setattr(self, key, value)

        self.tmax, self.mmax = self.model_pars(Sigma=self.Sigma, MGC0=self.MGC0, SFE=self.SFE)
        if (self.verbose):
            print(" t_max = %5.2f Myr"%self.tmax)
        if (self.tmax>self.tnuc):
            print(" Warning: tmax = %5.2f Myr > tnuc"%self.tmax)

        tmp, tmp, Mlow_frac = self.chabrier(m3=self.mmax)
        self.Mlow = Mlow_frac*self.MGC0

        self.mdw = self.mdwc*(10**self.FEH)**0.6
        
        # output times
        self.t = numpy.linspace(0,self.tmax,self.nt)
        self.Nmc = int(self.Nmc)
                
        # Derived quantities
        self.mdamax = (self.mmax-self.ms)/self.tmax

        # Initial abundances
        self.NaFe0 = self.initial_abundances[0]
        self.OFe0 = self.initial_abundances[1]
        self.MgFe0 = self.initial_abundances[1]
        self.AlFe0 = self.initial_abundances[2]

        
    def _initialise(self):
        """ create necessary arrays """
        numpy.random.seed(self.seed)
        
        # Discrete masses, evenly samples in log(mf)  
        self.mf = numpy.logspace(log10(self.ms),log10(self.mmax),self.Nmc)
        self.w = self.A*self.mf**(self.a+1)/self.Nmc *log(self.mmax/self.ms)
        
        # Mass bins
        self.lme =  numpy.linspace(log10(self.ms), log10(self.mmax), self.nm+1)
        self.me = 10**self.lme
        self.dm = self.me[1:] - self.me[0:-1]
        self.m = 10**(0.5*(self.lme[0:-1] + self.lme[1:]))
        self.mda = self.mdamax*( (self.mf-self.ms)/(self.mmax - self.ms))**0.5
        
        # Formation time
        self.tf = self.calc_tf(self.mf)
        tformmax = self.tmax-self.tf
        self.tform = self.sample_rate()*tformmax

        return 

    def _evolve(self):
        self.psi = []
        self.fdil = []
        self.mt = []
        self.Ym = []
        self.Tcm = []
        self.fdilm = []
        self.fdilm_c = []
        self.Ym_noacc = []
        self.DNm = []
        self.DNam = []
        self.DAlm = []
        self.xel = []
        self.dMdx = []
        self.dMdx_dil = []
        self.dMdx_dil_c = []
        self.Corr_dil = []
        self.psi_nowind = []
        self.mtmax = []
        self.Mwdot = []
        self.Mwdot_c = [] 
        self.mmw = []
        self.mYw = []
        self.mYw_noacc = []
        self.mwdot = []
        self.mwdot_c = []
        self.DMw = []
        self.Tc = []
        self.madot = []
        self.Madot = []
        self.Madoth = []
        self.Ndot = []
        self.SFR = []
        First = True
        
        for time in self.t:

            psi_nowind = self.calc_psi_nowind(time)
            self.psi_nowind.append(psi_nowind)

            psi,mt, madot  = self.calc_psi(time)
            mtmax = self.calc_m_accr(time, self.mdamax)

            self.psi.append(psi)
            self.mtmax.append(mtmax)

            mwdot = numpy.zeros_like(psi)
            mwdot_c = numpy.zeros((2,self.nm))
            
            c = (self.m>=self.mlim)&(self.m<mtmax)

            if (numpy.count_nonzero(c)>0):
                mwdot[c] = self.mdw*(self.m[c]/self.mw)**2 * psi[c]
                ct = (mt>=self.mlim)
                Mwdot = numpy.sum(self.w[ct]*self.mdw*(mt[ct]/self.mw)**2)

                age = time - self.tform
                tf = self.tf
                
                caems = (mt>=1000)&(age>=0)&(age<tf)
                cpvems = (mt>=100)&(age>=tf)

                Mwdot_c = []
                for j,cc in enumerate([caems, cpvems]):
                    Mwdot_c.append(numpy.sum(self.w[cc]*self.mdw*(mt[cc]/self.mw)**2))

                    for i in range(self.nm):
                        cm = (mt[cc]>self.me[i])&(mt[cc]<=self.me[i+1])
                        if numpy.count_nonzero(cm)>0:
                            mwdot_c[j,i] = numpy.sum(self.w[cc][cm]*self.mdw*(mt[cc][cm]/self.mw)**2)/self.dm[i]

                mmw = simpson(mwdot[c]*self.m[c],x=self.m[c])/Mwdot

                DMw = simpson(psi_nowind[c]*self.m[c], x=self.m[c]) -  simpson(psi[c]*self.m[c],x=self.m[c])
            else:
                Mwdot, Mwdot_c, mmw, DMw = 0, [0,0],0, 0

            Ndot = self.calc_Ndot(time)
            Madot = simpson(madot, x=self.m)

            c = (self.m>=self.mlim)
            Madoth = simpson(madot[c], x=self.m[c])
            SFR = self.Mlow/self.tmax + Madot + Ndot
            Mdot_low = self.Mlow/self.tmax 

            Ym, Ym_noacc, DNm,DNam, DAlm, xel, dMdx,dMdx_dil, dMdx_dil_c,Corr_dil,fdil,fdilm, fdilm_c = self.calc_abundances(time, mt,madot, mwdot, Mdot_low)

            c = (self.m>self.mlim)&(self.m<mtmax)
            Tcm = numpy.zeros_like(Ym)
            Tcm[c]  = self.M2T(self.m[c], Ym[c])
            if (numpy.count_nonzero(c)>1):
                mYw = simpson(mwdot[c]*Ym[c],x=self.m[c])/Mwdot
                mYw_noacc = simpson(mwdot[c]*Ym_noacc[c],x=self.m[c])/Mwdot
                tc = simpson(mwdot[c]*Tcm[c],x=self.m[c])/Mwdot
            else:
                mYw, mYw_noacc, tc = 0.25, 0.25, 0

            self.mt.append(mt)
            self.SFR.append(SFR)
            self.fdil.append(fdil)
            self.fdilm.append(fdilm)
            self.fdilm_c.append(fdilm_c)            
            self.Ym.append(Ym)
            self.Ym_noacc.append(Ym_noacc)
            self.Tcm.append(Tcm)
            self.DNm.append(DNm)
            self.DNam.append(DNam)
            self.DAlm.append(DAlm)
            if (First):
                self.xel = xel
                First = False
            
            self.dMdx.append(dMdx)
            self.dMdx_dil.append(dMdx_dil)
            self.dMdx_dil_c.append(dMdx_dil_c)
            self.Corr_dil.append(Corr_dil)
                
            self.Mwdot.append(Mwdot)
            self.Mwdot_c.append(Mwdot_c)
            self.mmw.append(mmw)
            self.Tc.append(tc)
            self.mYw.append(mYw)
            self.mYw_noacc.append(mYw_noacc)
            self.DMw.append(DMw)

            self.mwdot.append(mwdot)
            self.mwdot_c.append(mwdot_c)
            self.madot.append(madot)
            self.Madot.append(Madot)
            self.Madoth.append(Madoth)            
            self.Ndot.append(Ndot)

            
        self.psi = numpy.array(self.psi)
        self.psi_nowind = numpy.array(self.psi_nowind)
        self.mtmax = numpy.array(self.mtmax)
        self.mt = numpy.array(self.mt)
        self.SFR = numpy.array(self.SFR)

        self.Mwdot = numpy.array(self.Mwdot)
        self.Mwdot_c = numpy.array(self.Mwdot_c)
        self.mmw = numpy.array(self.mmw)
        self.mwdot = numpy.array(self.mwdot)
        self.Ym = numpy.array(self.Ym)
        self.Tcm = numpy.array(self.Tcm)
        self.DNm = numpy.array(self.DNm)
        self.DNam = numpy.array(self.DNam)
        self.DAlm = numpy.array(self.DAlm)
        self.dMdx = numpy.array(self.dMdx)
        self.dMdx_dil = numpy.array(self.dMdx_dil)
        self.dMdx_dil_c = numpy.array(self.dMdx_dil_c)
        self.Corr_dil = numpy.array(self.Corr_dil)
        self.fdil = numpy.array(self.fdil)
        self.fdilm = numpy.array(self.fdilm)
        self.fdilm_c = numpy.array(self.fdilm_c)

        self.mYw = numpy.array(self.mYw)
        self.mYw_noacc = numpy.array(self.mYw_noacc)
        self.DMw = numpy.array(self.DMw)
        self.DMwmax = self.A/(self.a+2) * (self.mmax**(self.a+2) - self.mw**(self.a+2))
        self.madot = numpy.array(self.madot)
        self.Madot = numpy.array(self.Madot)
        self.Madoth = numpy.array(self.Madoth)
        self.Ndot = numpy.array(self.Ndot)

        if len(self.Mwdot)>0:
            self.Mwind = simpson(self.Mwdot, x=self.t)
            self.Mwind_poll = simpson(numpy.sum(self.Mwdot_c, axis=1), x=self.t)
        # End init
        return

    def sample_rate(self):
        """ SFR: constant for now """ 
        # function describing the functional form
        # p = 2t => R=t**2 => t = R**0.5
        R = numpy.random.rand(self.Nmc)
        return R #**(1./(self.sfr_index+1))

    def model_pars(self, Sigma=1.5e3, MGC0=6e5, SFE=0.1):
        G = 0.004499
    
        # fixed parameters
        eps = 2.5e-3

        # compute cloud properties
        Mg = MGC0/SFE
        Rg = sqrt(Mg/(pi*Sigma))
        vrms = sqrt(3*G*Mg/(5*Rg))
        tau = Rg/vrms  
        mfmax = eps*Mg
        mdamax = mfmax/tau
        return tau, mfmax
    
    def sech(self, x):
        """ sech(x) function """
        xcrit = 10 # for x>xcrit approximate 1/cosh(x) as 2*exp(-x)
        if not hasattr(x,"__len__"):
            # x = scalar
            return 1/cosh(x) if x < xcrit else 2*exp(-x)
        else:
            # x = array x
            c = (x<xcrit)
            sech_func = numpy.zeros_like(x)
            sech_func[c] = 1/cosh(x[c])
            sech_func[~c] = 2*exp(-x[~c])
            return sech_func

    def _set_mf(self):
        # Maccr = Mtot - ms*N
        # Maccr = A
        a1 = self.a + 1
        a2 = self.a + 2
        ms, mmax = self.ms, self.mmax
        
        # high res above 100 Msun
        nm1 = 20
        nm2 = self.nm - nm1
        me1 = numpy.logspace(log10(ms), 1.99999, nm1+1)
        me2 = numpy.logspace(2, log10(0.999999*mmax), nm2)
        
        me = numpy.r_[me1, me2]

        self.m = 0.5*(me[0:-1] + me[1:])
        
        if self.Mlow == 0:
            self.A = self.Maccr/( (mmax**a2 - ms**a2)/a2 - ms*(mmax**a1-ms**a1)/a1)
            self.N = self.A/a1 * (mmax**a1 - ms**a1)
        else:
            s = 0.69
            mu = 0.08
            a = 1/(2*log(10)**2*s**2)
            m1 = 0.1
            Int = exp(0.25/a)*mu*sqrt(pi)/(2*sqrt(a))*( erf( (2*a*log(ms/mu)-1)/(2*sqrt(a))) - erf( (2*a*log(m1/mu)-1)/(2*sqrt(a))) ) 
            A1 = self.Mlow/Int
            self.A = A1*exp(-log10(ms/mu)**2/(2*s**2))/ms /ms**self.a
            self.N = self.A/a1 * (mmax**a1 - ms**a1)
            self.Maccr = self.A*( (mmax**a2 - ms**a2)/a2 - ms*(mmax**a1-ms**a1)/a1)
            self.Maccrh = self.A*( (mmax**a2 - self.mlim**a2)/a2 -self.mlim*(mmax**a1-self.mlim**a1)/a1)
        self.Mhigh = self.Maccr + self.N*self.ms
        return

    def calc_mdot_wind(self, m):
        return self.mdw*(m/self.mw)**2
    
    def calc_mda(self,mf):
        return self.mdamax*( (mf-self.ms)/(self.mmax-self.ms))**0.5  

    def calc_tf(self,mf):
        mda = self.calc_mda(mf)

        tf = numpy.zeros_like(mf) 
        c = (mda > 0)
        tf[c] = (mf[c]-self.ms)/mda[c]

        return  tf

    def calc_Ndot(self, t):
        # get the number of formation rate at time t
        Ndot = numpy.zeros_like(self.t)

        def Ndotm(mf):
            ndot = self.A*mf**self.a
            ndot *= 1/(self.tmax*(1 - sqrt((mf-self.ms)/(self.mmax-self.ms))))

            return ndot

        mtmax = ((self.tmax - t)/self.tmax)**2*(self.mmax-self.ms) + self.ms
        # careful with integrating to mmax
        if t==0: mtmax*= (1-SMALLNUMBER)
        Ndot = quad(Ndotm, self.ms, mtmax)[0]
        return Ndot

    def calc_psi_nowind(self,t):
        # evolve
        mt = numpy.zeros(self.Nmc)
        age = t-self.tform

        cacc = (age>=0)&(age<self.tf)
        cdone = (age>=self.tf)
        if numpy.count_nonzero(cacc)>0:
            mt[cacc] = self.ms + self.mda[cacc]*age[cacc]        
        if numpy.count_nonzero(cdone)>0:
            mt[cdone] = self.mf[cdone] # + self.mda[cacc]*t        

        c = (mt > 0)
        his, xm = numpy.histogram(log10(mt[c]), weights = self.w[c], bins=self.lme)
        return his/self.dm

    def calc_m_accr(self, age, mda):
        mw, mdw, ms = self.mw, self.mdw, self.ms
        # return mw*sqrt(mda/mdw)*tanh( sqrt(mda*mdw)*age/mw + arctanh(sqrt(mdw/mda)*ms/mw) ) 
        minf = mw*sqrt(mda/mdw)
        return minf*tanh( mda*age/minf + arctanh(ms/minf) ) 

    def calc_m_done(self, age, mda,tf):
        c = (mda>0)
        mtf = numpy.zeros_like(mda)
        
        mtf[c] = self.calc_m_accr(tf[c], mda[c])
        mw, mdw, ms = self.mw, self.mdw, self.ms
        return mtf/(1+mtf*mdw/mw**2*(age-tf))
        
    def calc_m(self, t):
        # calculate mass, works on arrays
        age = t-self.tform
        
        mt = numpy.zeros(self.Nmc)
        tf, mda = self.tf, self.mda
        
        caccr = (age>0)&(age<self.tf)
        mt[caccr] = self.calc_m_accr(age[caccr], mda[caccr])
    
        cdone = (age>=self.tf)
        mt[cdone] = self.calc_m_done(age[cdone], mda[cdone],tf[cdone])
        return mt

    def Ydot(self, t, Y, tform,tf,mda):

        tnuc = self.tnuc 
        ms, mw, mdw = self.ms, self.mw, self.mdw

        #mtmax = self.calc_m_accr(self.tmax, self.mdamax)
        mt = numpy.zeros_like(Y)
        age = t - tform
        
        caccr = (age>=0)&(age<tf)
        cdone = (age>=tf)

        if numpy.sum(caccr)>0:
            mt[caccr] = self.calc_m_accr(age[caccr], mda[caccr])
    
        if numpy.sum(cdone)>0:
            mt[cdone] = self.calc_m_done(age[cdone], mda[cdone],tf[cdone])

        Ydot = numpy.zeros_like(Y)
        c = (Y<1)&(mt>100)&(t>tform) #&(mt<mtmax)
        Ydot[c] = 0.75/tnuc 

        c = (mt>100)&(t>tform)&(age<tf) #&(mt<mtmax)
        Ydot[c] += (0.25-Y[c])*mda[c]/mt[c]

        return numpy.array(Ydot)

    def calc_Y(self, t, mt):

        Y = numpy.zeros_like(mt)+0.25
        Y_noacc = numpy.zeros_like(mt)+0.25

        mtmax = self.calc_m_accr(self.tmax, self.mdamax) # should be t?
        # only evolve stars aove 100 Msun
        cw = (mt>self.mw)&(mt<mtmax)
        
        if numpy.sum(cw)>0:
            mdavms = self.mda[cw]
            tformvms = self.tform[cw]
            tfvms = self.tf[cw]
            mtvms = mt[cw]
            Y0 = Y[cw]
            # solve odes for Y
            sol = ode(self.Ydot)
            sol.set_integrator('dopri5',atol=1e-6,rtol=1e-6,nsteps=1e7)
            sol.set_f_params(tformvms,tfvms,mdavms)
            sol.set_initial_value(Y0,0)

            sol.integrate(t)
            Y[cw] = sol.y

            age = t - tformvms
            Y_noacc[cw] += 0.75*age/self.tnuc
            Y_noacc[(Y_noacc>1)]=1
        return Y, Y_noacc

    def calc_maccr(self, t):

        maccr = numpy.zeros(self.Nmc)
        age = t - self.tform
        #age = t - self.tform
        
        caccr = (age>0)&(age<=self.tf)

        if numpy.count_nonzero(caccr)>0:
            maccr[caccr] = self.calc_mda(self.mf[caccr])
        return maccr

    def chabrier(self, m1=0.1, m3=1000, a=-2.3):
        """ returns Chabrier mass function """
        def func(m,index):
            # Chabrier IMF
            return m**index*exp(-log10(m/mu)**2/(2*s**2))/m

        m2 = 1
        s = 0.69
        mu = 0.08

        m = numpy.logspace(log10(m1), log10(m3),1000)
        psi = numpy.zeros_like(m)

        # constants of integration
        p2 = erf(log10(m2/mu)/(sqrt(2)*s))
        p1 = erf(log10(m1/mu)/(sqrt(2)*s))
        A1 = (sqrt(pi/2)*s*log(10)*(p2 - p1))**(-1)
        A2 = A1*exp(-log10(m2/mu)**2/(2*s**2))
        Int = 1 + A2*(m3**(a+1) - m2**(a+1))/(a+1)

        # normalise to 1 star
        A1/=Int
        A2/=Int
        
        # calculate mean mass
        Mlow = A1*quad(func, 0.1, 1, args=(1,))[0]
        Mhigh = A2*(m3**(a+2) - m2**(a+2))/(a+2)
        mmean = Mlow + Mhigh

        # Fraction of total mass in stars < 1 Msun
        Mlow_frac = Mlow/mmean
    
        c = (m<=m2)
        
        psi[c] = A1*func(m[c],0)
        psi[~c] = A2*m[~c]**a

        return m, psi, Mlow_frac
    
    def M2T(self, m, DY):
        """ returns Tc(m, Delta Y) """
        return (62.1 + 13.6*DY) + (10.02 + 5.85*DY)*(log10(m) - 3)
    
    def get_nucleo(self, mt, Y,Ynor,tt):
        c = (mt>self.mlim)
        
        FN, XN = 2*[numpy.zeros_like(mt)]
        FNa, XNa = 2*[numpy.zeros_like(mt)]
        FO, XO = 2*[numpy.zeros_like(mt)]
        FAl, XAl = 2*[numpy.zeros_like(mt)]
        FMg, XMg = 2*[numpy.zeros_like(mt)]

        if numpy.count_nonzero(c)>0:
            # Use the actual Y for Tc
            # use shorter Tc and DYnor
            Tc = self.M2T(mt[c], Y[c]-0.25)
            
            # And no rej for the abundances
            DYnor = Ynor[c]-0.25
        
            # From Prantzos+ 2007 (not 2017!)
            N0 = 8.482e-4 * 10**-1.5
            O0 = 5.4e-4
            Na0 = 6.9e-7 
            Mg0 = 4.7e-5
            Al0 = 2.0e-6

            T1 = 5*numpy.array(Tc/5, dtype='int')
            T2 = 5*(numpy.array(Tc/5, dtype='int')+1)
            
            DY12 = numpy.zeros((2,len(Tc)))
            N12 = numpy.zeros((2,len(Tc)))
            O12 = numpy.zeros((2,len(Tc)))
            Na12 = numpy.zeros((2,len(Tc)))
            Mg12 = numpy.zeros((2,len(Tc)))
            Al12 = numpy.zeros((2,len(Tc)))

            Tmod = numpy.array([50,55,60,65,70,75,80,85])
        
            d = numpy.zeros((len(Tmod), 32, 40))
            for i in range(len(Tmod)):
                file = "data_prantzos_etal_2017/nuc1_%03da.res"%Tmod[i]
                d[i]  = numpy.loadtxt(file,skiprows=1).T
            
            id1 = numpy.array( (T1-50)/5, dtype='int')

            for j in range(len(Tc)):
                for i in [0,1]:
                    time=d[id1[j]+i][1]
                    X   =d[id1[j]+i][4]
                    N   =d[id1[j]+i][10]
                    O   =d[id1[j]+i][11]
                    Na  =d[id1[j]+i][16]
                    Mg  =d[id1[j]+i][17]
                    Al  =d[id1[j]+i][21]            
                
                    DYmod = X[0]-X

                    N12[i][j] = numpy.interp(DYnor[j], DYmod, N/N0)
                    O12[i][j] = numpy.interp(DYnor[j], DYmod, O/O0)
                    Na12[i][j] = numpy.interp(DYnor[j], DYmod, Na/Na0)
                    Mg12[i][j] = numpy.interp(DYnor[j], DYmod, Mg/Mg0)
                    Al12[i][j] = numpy.interp(DYnor[j], DYmod, Al/Al0)

            FN[c] = (Tc - T1)/(T2-T1) * (N12[1] - N12[0]) + N12[0]
            FO[c]= (Tc - T1)/(T2-T1) * (O12[1] - O12[0]) + O12[0]
            FNa[c] = (Tc - T1)/(T2-T1) * (Na12[1] - Na12[0]) + Na12[0]
            FMg[c] = (Tc - T1)/(T2-T1) * (Mg12[1] - Mg12[0]) + Mg12[0]
            FAl[c] = (Tc - T1)/(T2-T1) * (Al12[1] - Al12[0]) + Al12[0]        

            # correct for initial abundances
            XN[c] = FN[c] # Prantzos value for [N/Fe] = 0
            XN[c] *= 10**(self.OFe0-0.4) # assume [O/Fe] = [alpha/Fe] and final N depends on O_0

            XNa[c] = FNa[c] # Prantzos value for [Na/Fe] = 0, [Ne/Fe] = = 0.4
            XNa[c] *= 10**(self.OFe0-0.4) # assume [Ne/Fe] = [alpha/Fe] 

            XO[c] = FO[c]*10**0.4          # Final oxygen independent of [alpha/Fe]
            XMg[c] = FMg[c]*10**self.MgFe0 # Mg depletion factor preserved

            XAl[c] = FAl[c]*10**(self.MgFe0-0.4) # More Al if more Mg
            
        return XN, XO,XNa,XMg,XAl

    def _clean_up(self):
        # Remove individual stellar values to save space:
        if (self.do_clean):
            self.tform = None
            self.tf = None
            self.mf = None
            self.mt = None
            self.w = None
            self.mda = None

    def calc_psi(self,t):
        # evolve all masses to time t in 1 go

        mt = numpy.array(self.calc_m(t))
        maccr = self.calc_maccr(t)
        
        c = (mt>0)
        his, xm = numpy.histogram(log10(mt[c]), weights = self.w[c], bins=self.lme)
        psi = 1.0*his/self.dm

        # short array len(psi)
        madot = numpy.zeros_like(psi)
        
        for i in range(self.nm):
            c = (mt>self.me[i])&(mt<=self.me[i+1])
            if numpy.count_nonzero(c)>0:
                madot[i] = numpy.sum(maccr[c]*self.w[c])/self.dm[i]

        return psi, mt, madot

    def calc_abundances(self,t,mt, madot,mwdot,Mdot_low):
        
        # hardcode for now
        ALi0 = 2.75
        
        # long array Nmc
        Y, Y_noacc = self.calc_Y(t, mt)

        XN,XO,XNa,XMg,XAl = self.get_nucleo(mt,Y,Y,t) 

        # short array len(psi)
        Ym = numpy.zeros_like(self.m) + 0.25
        Ym_noacc = numpy.zeros_like(self.m) + 0.25
        Nm = numpy.zeros_like(self.m)
        Nam = numpy.zeros_like(self.m)
        Alm = numpy.zeros_like(self.m) 
        
        for i in range(self.nm):
            c = (mt>self.me[i])&(mt<=self.me[i+1])
            if numpy.sum(c)>0:
                Ym[i] = numpy.sum(Y[c]*self.w[c])/numpy.sum(self.w[c])
                Ym_noacc[i] = numpy.sum(Y_noacc[c]*self.w[c])/numpy.sum(self.w[c])
                Nm[i] = numpy.sum(XN[c]*self.w[c])/numpy.sum(self.w[c])
                Nam[i] = numpy.sum(XNa[c]*self.w[c])/numpy.sum(self.w[c])
                Alm[i] = numpy.sum(XAl[c]*self.w[c])/numpy.sum(self.w[c])

        # element histogram DY, F_Na, F_Al
        nbin = 75
        hb = 1./30/2. # little offset to make sure 0 is incl as a mid bin
        
        Nel,Ncorr = 7, 2

        fdilm = numpy.zeros(self.nm)
        xel = numpy.zeros((Nel,nbin))
        dMdx = numpy.zeros((Nel,nbin))
        dMdx_dil = numpy.zeros((Nel,nbin))
        dMdx_dil_c = numpy.zeros((2,Nel,nbin))

        Corr_dil = numpy.zeros((Ncorr,nbin,nbin))
        Corr_dil_P1 = numpy.zeros((Ncorr,nbin,nbin))
        Corr_tmp = numpy.zeros((nbin,nbin))
        
        #       O-Na   Mg-Al 
        Cid = ( (3,4), (5,6) ) #, (4,2), (3,2), (4,6),(3,6) )
        self.Cid = Cid
        xrange = ( (-3,0), (0.+hb,2.5+hb), (-0.8+hb, 1.7+hb), (-1.7-hb,0.8-hb), (-0.8+hb, 1.7+hb), (-1.7-hb,0.8-hb), (-0.8+hb,1.7+hb))

        # Undiluted abundances
        X = numpy.array([Y-0.25, Y*0, XN, XO, XNa, XMg, XAl])

        # Pristine (log) abundances
        logX_prist=numpy.array([-numpy.inf,ALi0,0,self.OFe0,self.NaFe0,self.MgFe0,self.AlFe0])
        X_prist=10**logX_prist

        dt = self.tmax/(self.nt-1)

        # first select all stars:
        m1 = 100
        m2 = 1000
        age = t - self.tform

        # Select aEMS and pV/EMS
        caems = (mt>m2)&(age>0)&(age<=self.tf)
        cpvems = (mt>m1)&(age>self.tf)
        cpoll =  caems | cpvems

        fdilm_c = numpy.zeros((2,self.nm))
        
        # No winds
        idx = numpy.zeros(Nel,dtype='int')
        if numpy.count_nonzero(cpoll)==0: 
            # 1D histogram
            for i in range(Nel):
                bw = numpy.diff(xrange[i])/nbin
                xe = numpy.linspace(xrange[i][0], xrange[i][1], nbin+1)
                xel[i] = 0.5*(xe[1:] + xe[0:-1])
                idx[i] = (numpy.abs(xel[i] - logX_prist[i])).argmin()
                dMdx_dil[i][idx[i]] = Mdot_low/bw * dt
                dMdx_dil_c[0][i][idx[i]] = Mdot_low/bw * dt

            # Anti correlations
            for i in range(Ncorr):
                i1, i2 = Cid[i][0], Cid[i][1]
                idx1, idx2 = idx[i1], idx[i2]
                Corr_dil[i,idx1,idx2] = Mdot_low/bw**2 * dt
                Corr_dil[i] = Corr_dil[i].T
            mfdil = 1e3
            fdilm += 1e3
            fdilm_c += 1e3
            
        else:
            # Dilution

            # Select on properties of stars contribution: aEMS & pVEMS
            fdil = numpy.zeros_like(mt)

            ma_ = numpy.zeros_like(mt)
            mw_ = numpy.zeros_like(mt)

            mdw = self.calc_mdot_wind(mt) # compute once

            twopoll = False
            if (twopoll):
                # First aEMS
                fdil[caems] = self.mda[caems]/mdw[caems]
                ma_[caems] = self.mda[caems]
                mw_[caems] = mdw[caems]
            
                # Now the stars that are done: pVEMS
                Madot_aEMS = sum(self.w[caems]*self.mda[caems])
                Mwdot_pVEMS = sum(self.w[cpvems]*mdw[cpvems])
                Mwdot = sum(self.w[cpoll]*mdw[cpoll])
            
                # provide an average accretion on pVEMS to make the budget
                N_pVEMS = sum(self.w[cpvems])
                mda_pVEMS = (Mdot_low - Mwdot - Madot_aEMS)/N_pVEMS
                ma_[cpvems] = mda_pVEMS
                mw_[cpvems] = mdw[cpvems]

                fdil[cpvems] = ma_[cpvems]/mw_[cpvems]

            # Simpler dilution model
            Npoll = sum(self.w[cpoll])
            Mwdot = sum(self.w[cpoll]*mdw[cpoll])
            ma_[cpoll] = (Mdot_low - Mwdot)/Npoll
            mw_[cpoll] = mdw[cpoll]
            fdil[cpoll] = ma_[cpoll]/mw_[cpoll] 
            
            mfdil = numpy.sum(ma_[cpoll]*self.w[cpoll])/numpy.sum(mw_[cpoll]*self.w[cpoll])

            mfdil_check = (Mdot_low - Mwdot)/Mwdot
            c = cpoll
            if (self.verbose):
                print(" t = %3.1f;  min(fdil) = %8.2f; max(fdil) = %9.2f; <fdil> = %9.2f, %9.2f; DY = %5.2f"%(t, min(fdil[c]),max(fdil[c]), mfdil, mfdil_check, max(Y)-0.25))

            for j,c in enumerate([caems, cpvems]):
                for i in range(self.nm):
                    cm = (mt[c]>self.me[i])&(mt[c]<=self.me[i+1])
                    if numpy.count_nonzero(cm)>0:
                        fdilm_c[j,i] = numpy.sum(fdil[c][cm]*self.w[c][cm])/numpy.sum(self.w[c][cm]) 

            for i in range(self.nm):
                cm = (mt[c]>self.me[i])&(mt[c]<=self.me[i+1])
                if numpy.count_nonzero(cm)>0:
                    fdilm[i] = numpy.sum(fdil[c][cm]*self.w[c][cm])/numpy.sum(self.w[c][cm]) 

           # fdil = fdil[cpoll] # TMP
            X_dil = numpy.zeros((Nel,numpy.count_nonzero(cpoll)))
            
            for i in range(Nel):
                bw = numpy.diff(xrange[i])/nbin
                xe = numpy.linspace(xrange[i][0], xrange[i][1], nbin+1)
                xel[i] = 0.5*(xe[1:] + xe[0:-1])
                idx[i] = (numpy.abs(xel[i] - logX_prist[i])).argmin()
                
                if i!=1:
                    dMdx[i], xele  = numpy.histogram(log10(X[i][cpoll]), weights=self.w[cpoll]*mdw[cpoll]/bw*dt,range=xrange[i], bins=nbin)


                # abundances
                for j,c in enumerate([caems, cpvems, cpoll]):
                    X_dil_ = (X[i][c] + fdil[c]*X_prist[i])/(1 + fdil[c])
                    dMdx_dil_, xele  = numpy.histogram(log10(X_dil_), weights=self.w[c]*mdw[c]*(1+fdil[c])/bw*dt,range=xrange[i], bins=nbin)

                    if j<2:
                        dMdx_dil_c[j][i] = dMdx_dil_
                    else:
                        dMdx_dil[i] = dMdx_dil_
                        
                X_dil[i] = X_dil_

            # Anti corrs
            for i in range(Ncorr):
                i1, i2 = Cid[i][0], Cid[i][1]
                Corr_dil[i],crap, crap= numpy.histogram2d(log10(X_dil[i1]),log10(X_dil[i2]), weights=self.w[cpoll]*mdw[cpoll]*(1+fdil[cpoll])/bw**2*dt,bins=nbin,range=(xrange[i1], xrange[i2]))

                # Add P1
                i1, i2 = Cid[i][0], Cid[i][1]
                idx1, idx2 = idx[i1], idx[i2]
                Corr_dil[i] = Corr_dil[i].T


        return Ym, Ym_noacc, Nm, Nam, Alm, xel,dMdx,dMdx_dil,dMdx_dil_c,Corr_dil, mfdil, fdilm, fdilm_c

    

