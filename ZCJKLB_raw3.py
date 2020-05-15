## If you want to simulate solar cell under illumination, you require
## an additional exciton generation profile that has the same number of
## points as L+1 (number of slices)

# Author: Liu Bo/Jun Kai/Zhao CHao
# Edited by: Zong Long
# Version 1.0.0.1

# Changelog:
# Debugging, adding and removing print commands. Added timer feature.
# Removing extractor feature.



import os
import sys
from scipy.special import jn
from numpy import *
import numpy as np
import matplotlib.pyplot as plt  #
from math import *
#import math
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.linalg  import *  # spsolve
from scipy.sparse.linalg import spsolve
from scipy.interpolate import UnivariateSpline
import smtplib
from datetime import datetime

B=1       ##ideality breakdown factor
Q= 1.6022e-19   ## Elementary charge in Coulomb
kB=1.38066e-23  ## Boltzmann constant
PI = 3.1416
T=300   ## Temperature in Kelvins
Vt=B*kB*T/Q     
DC=8.854e-12  ## Permittivity of free space
Dielectric_Constant =3*DC
up_main=1e-7     ## in m^2/Vs, hole mobility
un_main=1e-7     ## in m^2/Vs, electron mobility
Egap=1.15  ## HOMO-LUMO, exciton binding energy is already taken into consideration
V0=0.73   ## Bare potential:
ni=7e26*exp(-Egap/2/Vt)   ## Fictitious parameter consisting of intrinsic value of charges inside
                          ## 7e26 calculated from P3HT (5e26 m-3) and PCBM (1e27 m-3)
aa=2.8      ## e-h distance in CT state, in nm        
kf=1e5      ## geminate recombination rate constant, in s^-1        
gamma=0.1   ## recombination control coefficient: 1 for Langevin
Gamma =gamma*Q/Dielectric_Constant *min(up_main,un_main)
StoredGamma=Gamma
VaStep=0.07  ## voltage step for JV curve:
             ## choose a VaStep such that V[0]-V[L] will never be zero
jpoints=21  ## number of points to simulate, including Va=0
L = 415   ## number of points in PAL
h=2.0e-10   ## step size per point in PAL in m
sun=[20]    ## input list for number of suns
PL_list=[2e23]  ## input list for hole contact carrier densities
N0_list=[2e23]  ## input list for electron contact carrier densities

CMCSrange_n=3      ## number of points used to extrapolate n[L]
CMCSrange_p=3      ## number of points used to extrapolate p[0]

reset=0

## Initialise mobility
up=array([up_main for i in xrange(0,L+1)])  
un=array([un_main for i in xrange(0,L+1)])  



def first_derivative_V(m):
    d_m=[0.0 for i in xrange(0,L+1)]
    m_half=[0.0 for i in xrange(0,L)]
    for i in xrange (0,L):
        m_half[i]=(m[i+1]+m[i])/2
    for i in xrange (1,L):
        d_m[i]= (m_half[i]-m_half[i-1])/h
    d_m[L]=d_m[L-1]
    d_m[0]=d_m[1]
    return d_m

def spl_half(m):
    x_f=[(i+0.5) for i in xrange(0,L)]
    x_i=[i for i in xrange(0,L+1)]
    y_f=[0.0 for i in xrange(0,L)]
    s_i=UnivariateSpline(x_i,m,s=0)
    s_f=s_i(x_f)
    return s_f

def spl_dev(m):
    x_if=[i for i in xrange(0,L+1)]
    s_i=UnivariateSpline(x_if,m,s=0)
    ds_i=s_i.derivative()
    ds_f=ds_i(x_if)
    return ds_f
    
def second_derivative(m):
    m1=[0.0 for i in xrange(0,L+1)]
    for i in range (1,L):
        m1[i]=(m[i+1]-2*m[i]+m[i-1])/h/h
    m1[L]=m1[L-1]
    m1[0]=m1[1]
    return m1

# linear function for fitting purposes
def func(x,a,b):
    return a*x+b


# log the carrier densities, fit linearly then determine n @ boundary
def CMCS_n(m):
    x_i=[i for i in xrange(0,CMCSrange_n)]
    y_i=[0 for i in xrange(0,CMCSrange_n)]
    for i in xrange(0,CMCSrange_n):
        y_i[i]=log10(m[i])
    par,pcov=curve_fit(func,x_i,y_i)
    extpt=10**func(CMCSrange_n,par[0],par[1])
    return extpt
    
def CMCS_p(m):
    x_i=[i for i in xrange(0,CMCSrange_p)]
    y_i=[0 for i in xrange(0,CMCSrange_p)]
    for i in xrange(0,CMCSrange_p):
        y_i[i]=log10(m[i])
    par,pcov=curve_fit(func,x_i,y_i)
    extpt=10**func(-1,par[0],par[1])
    return extpt    

#b
def b(E_Field,T):
    return Q**3*E_Field/8/PI/Dielectric_Constant/kB**2/T**2


# e-h pair binding energy
def Eb(e_h_distance):
    return Q**2/4/PI/Dielectric_Constant/e_h_distance*1e9

#dissociation constant
def k_diss(e_h_distance, E_Field, T):
    return 3*Gamma/gamma*e**(-Eb(e_h_distance)/kB/T)/4/PI/(aa*1e-9)**3*(jn(1,2*math.sqrt(2*b(E_Field,T))*1j)/(math.sqrt(2*b(E_Field,T))*1j)).real

#p for dissociation probability
def p5(e_h_distance, E_Field, T, aa, kff):
    return k_diss(e_h_distance, E_Field, T)/(k_diss(e_h_distance, E_Field, T) + kff)*4/math.sqrt(PI)/aa**3*e_h_distance*e_h_distance*exp(-e_h_distance*e_h_distance/aa/aa)

def poisson_npV(n,p,V, PHI_n,PHI_p):
    for t in xrange (1,30):        
        dd_V=second_derivative(V)
    
        M = mat([[0.0 for i in xrange(L+1)] for y in xrange(L+1)])
        for i in xrange (0,L):
            M[i,i]=-2.0/h/h-Q/Dielectric_Constant/Vt*(n[i]+p[i])      
            M[i+1,i]=1.0/h/h
            M[i,i+1]=1.0/h/h
        M[0,0]=1.0
        M[0,1]=0.0
        M[L,L-1]=0.0
        M[L,L]=1.0
        
        N = Q/Dielectric_Constant *(n-p)-dd_V 
        N[0]=0
        N[L]=0
        DELTA = spsolve(M,N)
    
        for i in xrange (0,L+1):
            if (DELTA[i]>Vt*1.5):
                V1[i]= V[i]+1.5*Vt
            elif (DELTA[i]<-Vt*1.5):
                V1[i]= V[i]-1.5*Vt
            else:
                V1[i]= V[i]+DELTA[i]

        dd_V1=second_derivative(V1)
        for i in xrange (0, L+1):
            
            n1[i]=n[i]*(1+DELTA[i]/Vt)
            p1[i]=p[i]*(1-DELTA[i]/Vt)

        if min(n1)<=0 or min(p1)<=0:
            print "charge carrier density break down in poisson at %s" %textname
            n=n1
            p=p1
            break
        V=V1
        n=n1
        p=p1

        if max(abs(DELTA)) < 0.00001: 
#            print 'poisson break' ,t
            break
    return n,p,V, PHI_n,PHI_p




def continuity(n,p,V, PHI_n,PHI_p,P,redo):     
    delta_n =array([0 for i in xrange(0,L+1)])
    delta_p =array([0 for i in xrange(0,L+1)])
    DELTA =array([0 for i in xrange(0,L+1)])
    a_n=array([1.0 for i in xrange(0,L+1)])
    a_p=array([1.0 for i in xrange(0,L+1)])
    d_PHI_n=array([1.0 for i in xrange(0,L+1)])
    d_PHI_p=array([1.0 for i in xrange(0,L+1)])
    d_a_n=array([1.0 for i in xrange(0,L+1)])
    d_a_p=array([1.0 for i in xrange(0,L+1)])
    N_n=array([0.0 for i in xrange(0,L+1)])  
    N_p=array([0.0 for i in xrange(0,L+1)])
    N1_n=array([0.0 for i in xrange(0,L+1)])  
    N1_p=array([0.0 for i in xrange(0,L+1)])
    U=array([0.0 for i in xrange(0,L+1)])
    
    a_n_half=array([1.0 for i in xrange(0,L)])
    a_p_half=array([1.0 for i in xrange(0,L)])
    Tol_p=array([1.0 for i in xrange(0,L+1)])
    Tol_n=array([1.0 for i in xrange(0,L+1)])
    Tol=array([0.0 for i in xrange(0,L+1)])

   
    for t2 in xrange (1,30):
        for i in xrange(0,L+1):
            a_n[i]=Vt*un[i]*ni*exp(V[i]/Vt)
            a_p[i]=Vt*up[i]*ni*exp(-V[i]/Vt)

####### n start

        a_n_half=spl_half(a_n)

        M_n = mat([[0.0 for i in xrange(L+1)] for y in xrange(L+1)])
        for i in xrange (1,L):
            M_n[i,i]=-a_n_half[i]-a_n_half[i-1]-(1-P[i])*Gamma*ni*ni*PHI_p[i]*h*h     
            M_n[i,i+1]=a_n_half[i]
            M_n[i,i-1]=a_n_half[i-1]
        M_n[0,0]=1.0
        M_n[0,1]=0.0    ## to prevent the first point from changing
        
        M_n[L,L-1]=0.0  ## to prevent the last point from changing
        M_n[L,L]=1.0


        for i in xrange (1,L):

            N_n[i]=(-P[i]*Gtemp[i]+(1-P[i])*Gamma*ni*ni*(PHI_n[i]*PHI_p[i]-1))*h*h+a_n_half[i-1]*(PHI_n[i]-PHI_n[i-1])+a_n_half[i]*(PHI_n[i]-PHI_n[i+1])

        N_n[0]=0
        N_n[L]=0

        delta_n =spsolve(M_n,N_n)


        
###### n end
###### p start

        a_p_half=spl_half(a_p)

        M_p = mat([[0.0 for i in range(L+1)] for y in range(L+1)])
        for i in xrange (1,L):
            M_p[i,i]=-a_p_half[i]-a_p_half[i-1]-(1-P[i])*Gamma*ni*ni*PHI_n[i]*h*h       
            M_p[i,i+1]=a_p_half[i]
            M_p[i,i-1]=a_p_half[i-1]
        M_p[0,0]=1.0
        M_p[0,1]=0.0
        
        M_p[L,L-1]=0.0
        M_p[L,L]=1.0


        for i in xrange (1,L):

            N_p[i]=(-P[i]*Gtemp[i]+(1-P[i])*Gamma*ni*ni*(PHI_n[i]*PHI_p[i]-1))*h*h+a_p_half[i-1]*(PHI_p[i]-PHI_p[i-1])+a_p_half[i]*(PHI_p[i]-PHI_p[i+1])

        N_p[0]=0
        N_p[L]=0

        delta_p =spsolve(M_p,N_p)

###### p end

        PHI_p=PHI_p+delta_p 
        PHI_n=PHI_n+delta_n   # update now to calc PHI_p and PHI_n

        if redo==0:
            PHI_p[0]=PHI_p[L]
            PHI_n[L]=PHI_n[0]
        
        for i in xrange (0,L+1):
            n1[i]=ni*exp((V[i]/Vt))*PHI_n[i]
            p1[i]=ni*exp((-V[i]/Vt))*PHI_p[i]
            

        if min(n1)<=0 or min(p1)<=0:
            print "charge carrier density break down in continuity at %s" %textname
            n=n1
            p=p1
            break
        
        n=n1
        p=p1

        for i in xrange (1,L+1):
            Tol[i]=(abs(delta_n[i])*ni*exp(V[i]/Vt)+abs(delta_p[i])*ni*exp(-V[i]/Vt))/(n1[i]+p1[i])

        if max(abs(Tol)) < 0.0001:
#            print 'continuity break', t2
            break

    return n,p,V, PHI_n,PHI_p


######## DEV PARAM CALCULATOR #########

def calcdp(volt,curr):
    pwr=[0.0 for x in range(0,len(volt))]
    vmax=0.0
    jmax=0.0
    jsc=curr[0]
    voc=0.0

    for i in xrange(0,len(volt)):
        pwr[i]=-volt[i]*curr[i]
    for i in xrange(0,len(volt)):        
        if pwr[i]==max(pwr):
            Vmax=volt[i]
            Jmax=curr[i]
    for i in xrange(0,len(volt)):
        if curr[i]>0:
            voc=(volt[i-1]*curr[i]-volt[i]*curr[i-1])/(curr[i]-curr[i-1])
            break
    for i in xrange(0,len(volt)):
        if volt[i]>0:
            jsc=(volt[i]*curr[i-1]-volt[i-1]*curr[i])/(volt[i]-volt[i-1])
            break
    FF=Jmax*Vmax/(voc*jsc)
    return voc,jsc,FF,Vmax,Jmax

######## EXTRACTOR #########

def extractor(foldername,NvRange,NcRange,sunname,filename): # reads data from previous point
    
    n_ex=array([0.0 for i in range(0,L+1)])
    p_ex=array([0.0 for i in range(0,L+1)])
    V_ex=array([0.0 for i in range(0,L+1)])
    PHI_n_ex=array([0.0 for i in range(0,L+1)])
    PHI_p_ex=array([0.0 for i in range(0,L+1)])
    f=open('%s/Np %s/Nn %s/Sun %s/%s-%sSun.txt' %(foldername,NvRange,NcRange,sunname,filename,sunname))      ## opens text file as f
    lines = f.readlines()       ## reads all lines

    for i in xrange(1,len(lines)):
        words = lines[i].split()
        n_ex[i-1]=float(words[3])
        p_ex[i-1]=float(words[4])
        PHI_n_ex[i-1]=float(words[7])
        PHI_p_ex[i-1]=float(words[8])
        V_ex[i-1]=float(words[9])
    f.close()

    return n_ex,p_ex,PHI_n_ex,PHI_p_ex,V_ex



filename3=os.path.basename(sys.argv[0])
filename3=filename3[:-3]
filelist = os.listdir(os.getcwd())  # working dir
#print os.path.exists(filename3)  #whether folder exist

basename = os.path.basename(os.getcwd())
#print basename
if os.path.exists(filename3)!=True:
    os.mkdir(filename3)

Gtemp=array([0.0 for i in range (0,L+1) ])



for ran1 in xrange(0,len(PL_list)):          

    if os.path.exists('%s/Np %s' %(filename3,PL_list[ran1]))!=True:
        os.mkdir('%s/Np %s' %(filename3,PL_list[ran1]))
    for ran2 in xrange(0,len(N0_list)):     

        devparam=array([[0.0 for i in range(5)] for j in range(len(sun))])
        
        if os.path.exists('%s/Np %s/Nn %s' %(filename3,PL_list[ran1],N0_list[ran2]))!=True:
            os.mkdir('%s/Np %s/Nn %s' %(filename3,PL_list[ran1],N0_list[ran2]))
        
        for lit in xrange(0,len(sun)):
            TimeStart=datetime.now()
            G1=array([0.0 for i in range (0,L+1) ])
            if sun[lit]!=0:
                f=open('83nm PAL abs1.txt')      #### YOU CAN CHANGE GEN PROFILE HERE ####
                                                 #### Need to have same no. of points as L ###
                lines=f.readlines()             #get a list with separate lines including '/n'

                for i in xrange (0,L+1):
                    generat = lines[i].split()
                    G1[i]=float(generat[0])
            

            Va=-0.07      ## Starting bias, applied on L side relative to 0 side
                                
            Va2=Va      ## Present bias, applied on L side relative to 0 side
                        ## Causes V@L to increase and V@0 to decrease by 0.5*Va2
            flip=0
            rep=0
            rd=0
            boom=0
            Ve1=V0-Va   ## Electrostatic potential difference (by convention refers to hole potential)
                        ## at 0 side relative to L side.
                        ## for the previous iteration 
            NL=0.0
            P0=0.0

            if os.path.exists('%s/Np %s/Nn %s/Sun %s' %(filename3,PL_list[ran1],N0_list[ran2],sun[lit]))!=True:
                os.mkdir('%s/Np %s/Nn %s/Sun %s' %(filename3,PL_list[ran1],N0_list[ran2],sun[lit]))
            

            for i in range(0,L+1):
                Gtemp[i]=sun[lit]*G1[i]


            ########### Declaration of Arrays #############

            x=array([1.0*i for i in xrange(0,L+1)]) #0 at cathode, L at anode 
            V=array([0.0 for i in xrange(0,L+1)])   
            n=array([0.0 for i in xrange(0,L+1)])
            p=array([0.0 for i in xrange(0,L+1)])

            CMCS_nspace=array([0.0 for i in xrange(0,CMCSrange_n)])
            CMCS_pspace=array([0.0 for i in xrange(0,CMCSrange_p)])

            V1=array([0.0 for i in xrange(0,L+1)])
            n1=array([0.0 for i in xrange(0,L+1)])
            p1=array([0.0 for i in xrange(0,L+1)])

            PHI_n=array([1.0 for i in xrange(0,L+1)])
            PHI_p=array([1.0 for i in xrange(0,L+1)])
            d_PHI_n=array([1.0 for i in xrange(0,L+1)])
            d_PHI_p=array([1.0 for i in xrange(0,L+1)])
            QF_n=array([0.0 for i in xrange(0,L+1)])
            QF_p=array([0.0 for i in xrange(0,L+1)])

            N0=N0_list[ran2]
            PL=PL_list[ran1]               
            for i in xrange (0,L+1):
                V[i]=(V0-Va)*(1.0/2-i*1.0/L)        ## input the starting V for no reference available
                n[i]=N0*exp(-V0*i/(L*Vt))           ## input the starting n for no reference available
                p[i]=PL*exp(-V0*(L-i)/(L*Vt)) ## input the starting p for no reference available
                #print -V0*(1.0/2-i*1.0/L)
                #print i,p[i]
                        

            UU=array([0.0 for i in xrange(0,L+1)])
            X=array([0.0 for i in xrange(0,L+1)])
            J_n =array([0.0 for i in xrange(0,L+1)])
            J_p =array([0.0 for i in xrange(0,L+1)])
            E =array([0.0 for i in xrange(0,L+1)])
            P =array([0.0 for i in xrange(0,L+1)])
            R =array([0.0 for i in xrange(0,L+1)])
            k_dissoc=array([0.0 for i in xrange(0,L+1)])

            n_ref= array([1.0 for i in xrange(0,L+1)])
            p_ref= array([1.0 for i in xrange(0,L+1)])
            V_ref= array([1.0 for i in xrange(0,L+1)])

            Jv = array([0.0 for i in xrange(0,jpoints)])
            jV = array([0.0 for i in xrange(0,jpoints)])
                            
            ############# POISSON CONTINUITY LOOP #############
                
            for vv in range(0,jpoints):
                
                Va2=Va+VaStep*vv
                if vv==0 and sun[lit]==0:
                    Gamma=0
                    print("debugcondition1", Ve1, Va2)
                

                if vv>0 and flip==0:
                    textname = "Jpoint-"+str(Va2-VaStep)
                    print(p[2],p[56],V[45])
                    n,p,PHI_n,PHI_p,V = extractor(filename3,PL_list[ran1],N0_list[ran2],sun[lit],textname)
                    print(p[2],p[56],V[45])
                    for i in xrange (0,L+1):    
                        V[i]=((Ve1-VaStep)/Ve1)*V[i]
                    Ve1-=VaStep
                    print("debugcondition2", Ve1, Va2)

                    
                if flip==1 :
                    textname = "Jpoint-"+str(Va2-VaStep)
                    print(p[2],p[56],V[45])
                    n,p,PHI_n,PHI_p,V = extractor(filename3,PL_list[ran1],N0_list[ran2],sun[lit],textname)
                    print(p[2],p[56],V[45])
                    for i in xrange (0,L+1): 
                        V[i] = V[i]-VaStep*(1.0/2-i*1.0/L)
                    Ve1-=VaStep
                    flip=0
                    rep=1
                    print("debugcondition3", Ve1, Va2)

                if Ve1<0 and Ve1>= -VaStep:
                    flip=1
                    print("debugcondition4", Ve1, Va2)
                    
                textname = "Jpoint-"+str(Va2)
                
#%%                
                for tt in xrange(1,300):
                    for i in xrange (0, L+1):  
                        n_ref[i]=n[i]   
                        p_ref[i]=p[i]
                        V_ref[i]=V[i]
#%%

            ### Applying Poisson engine ####
                
                    n,p,V, PHI_n,PHI_p= poisson_npV(n,p,V, PHI_n,PHI_p)

            ### Prob ###
#%%
                    E = first_derivative_V(V)
                    if reset==0:
                        for i in xrange(0,L+1):
                            def p_diss(e_h_distance):
                                return p5(e_h_distance, abs(E[i]), T, aa, kf)
                            P[i], err =integrate.quad(p_diss, 0, inf)             ## INTEGRATION TAKES HELL LONG TIME!!!
#                        print((P[0],P[1],P[2], rep, reset, rd, vv, tt))    
                        reset=1     # skips integration for following iterations

                    if rep==1:
                        reset=0
                        rep=0
                    if vv==0 and boom==0: ## to run probability integration again
                        reset=0
                        rep=1
                        boom=1
                    PHI_n[0]=N0/ni/exp(V[0]/Vt)
                    PHI_p[L]=PL/ni/exp(-V[L]/Vt)
                    if vv==0:
                        NL=ni*exp(V[L]/Vt)*PHI_n[0] 
                        P0=ni*exp(-V[0]/Vt)*PHI_p[L]
                        n[L]=NL
                        p[0]=P0

                    if vv>0:       ## to extrapolate the hole CCD at electron contact, and vice versa
                        ##for n
                        for i in xrange(0,CMCSrange_n):
                            CMCS_nspace[i]=n[i+L-CMCSrange_n]
                        n[L]=CMCS_n(CMCS_nspace)
                        ##for p
                        for i in xrange(0,CMCSrange_p):
                            CMCS_pspace[i]=p[i+1]
                        p[0]=CMCS_p(CMCS_pspace)

                    PHI_n[L]=n[L]/ni/exp(V[L]/Vt)
                    PHI_p[0]=p[0]/ni/exp(-V[0]/Vt)  ## update the dummy variable


#%%

            ### Applying Continuity engine ###

                    n,p,V, PHI_n,PHI_p= continuity(n,p,V, PHI_n,PHI_p, P,rd)  #G_470_140_80 add C-Dnp
                    
#                    print "round %s passed" %tt

                    if min(n)<=0 or min(p)<=0:
                        print "charge carrier density break down in %s" %textname
                        break



                    if rd==1:
                        if max(abs(n_ref*1.0/n-1.0))<0.01 and max(abs(p_ref*1.0/p-1.0))<0.01 and max(abs(V_ref-V))<Vt*0.02:    
                            print "tt=", tt
                            print "poisson and continuity satisfied"
                            if vv==0:
                                Gamma=StoredGamma
                            break

                    if rd==0:
                        if max(abs(n_ref*1.0/n-1.0))<0.01 and max(abs(p_ref*1.0/p-1.0))<0.01 and max(abs(V_ref-V))<Vt*0.02:    
                            print "tt=", tt
                            print "poisson and continuity satisfied, redo"
                            rd=1

                d_PHI_n=spl_dev(PHI_n)/h
                d_PHI_p=spl_dev(PHI_p)/h
                E= first_derivative_V(V)

                for i in range (0,L+1):
                    UU[i]=(P[i]*Gtemp[i])

                for i in xrange (0,L+1):    
                    J_n[i]=-Q*Vt*un[i]*ni*exp(V[i]/Vt)*d_PHI_n[i] 
                    J_p[i]=Q*Vt*up[i]*ni*exp(-V[i]/Vt)*d_PHI_p[i]
                    QF_n[i]= -Vt*log(PHI_n[i])
                    QF_p[i]= Vt*log(PHI_p[i])
                J_n[0]=J_n[1]
                J_p[0]=J_p[1]

                J_p[L]=J_p[L-1]
                J_n[L]=J_n[L-1]


                f = open('%s/Np %s/Nn %s/Sun %s/%s-%sSun.txt' %(filename3,PL_list[ran1],N0_list[ran2],sun[lit],textname,sun[lit]),'w')
                f.writelines("x[i] E[i](V/m) P[i] n[i](m-3) p[i](m-3) UU[i](m-3) G1[i](m-3) PHI_n[i] PHI_p[i] V[i](V) J_n[i](Am-2) J_p[i](Am-2) J(Am-2)\n")
                for i in xrange(0,L+1):

                    f.writelines("%g %g %g %g %g %g %g %g %g %g %g %g %g\n" %(x[i], E[i], P[i], n[i], p[i], UU[i], G1[i], PHI_n[i], PHI_p[i], V[i], J_n[i], J_p[i], J_n[i]+J_p[i]))

                f.close()

                Jv[vv]= (J_n[L/2]+J_p[L/2])/10       ##   mA cm-2, for plotting
                jV[vv]= Va2
#%%

            #################### PLOTTING STABLE STATE ####################


                fig2=plt.figure(figsize=(10, 6), dpi=100) 
                ax5=fig2.add_subplot(231)

                plt.plot(x, UU, label="UU")

                plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
                ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})

                ax6=fig2.add_subplot(232)
                plt.plot(x, QF_n, label="QF_n")
                plt.plot(x, QF_p, label="QF_p")

                plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
                ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})

                ax7=fig2.add_subplot(233)
                plt.plot(x, V, label="V")

                plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
                ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})

                ax8=fig2.add_subplot(234)

                plt.semilogy(x, n, label="n")
                plt.semilogy(x, p, label="p")

                plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
                ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})

                ax9=fig2.add_subplot(235)

                plt.plot(x, J_n, label="J_n")
                plt.plot(x, J_p, label="J_p")
                plt.plot(x, J_n+J_p, label="J")                
                plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
                ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})


                ax10=fig2.add_subplot(236)

                plt.plot(x, P, label="P")
                plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
                ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})


                plt.savefig('%s/Np %s/Nn %s/Sun %s/%s-%sSun.png' %(filename3,PL_list[ran1],N0_list[ran2],sun[lit],textname,sun[lit]))  

                plt.close()
                reset=0
            ##################  PLOTTING JV CURVE #######################


            fig3=plt.figure(figsize=(10, 6), dpi=100)

            plt.semilogy(jV, Jv, label="JV curve")
            plt.legend( bbox_to_anchor=(0.2, 1, 0.8, .9), loc=3,
            ncol=4,mode="expand", borderaxespad=0,prop={"size":"smaller"})

            plt.savefig('%s/Np %s/Nn %s/Sun %s/%s-%sSun.png' %(filename3,PL_list[ran1],N0_list[ran2],sun[lit], "JV Curve",sun[lit]))

            plt.close()


            ##########  WRITING JV CURVE POINTS ##########

            TimeEnd=datetime.now()

#            ts=datetime.datetime.fromtimestamp(TimeStart).strftime('%Y-%m-%d %H:%M:%S')
#            tf=datetime.datetime.fromtimestamp(TimeEnd).strftime('%Y-%m-%d %H:%M:%S')
#            timedelta = tf - ts


            f2 = open('%s/Np %s/Nn %s/Sun %s/%s-%sSun.txt' %(filename3,PL_list[ran1],N0_list[ran2],sun[lit], "JV Curve",sun[lit]), 'w')

#            f2.writelines("Program started: %s\n" %(ts))
#            f2.writelines("Program ended: %s\n" %(tf))

            f2.writelines("Va\tJ(mA cm-2)\n")
            for i in xrange(0,jpoints):
                f2.writelines("%g\t%g\n" %(jV[i], Jv[i]))
            f2.close()

        ######### WRITING DEVICE PARAMETERS #########
            devparam[lit,0], devparam[lit,1],devparam[lit,2],devparam[lit,3],devparam[lit,4]=calcdp(jV,Jv)
        f3 = open('%s/Np %s/Nn %s/%s-Nv%sNc%s.txt' %(filename3,PL_list[ran1],N0_list[ran2],"SolCellDevParam",PL_list[ran1],N0_list[ran2]), 'w')
        f3.writelines("Sun Voc Jsc FF Vmpp Jmpp\n")
        for i in xrange(0,len(sun)):
            f3.writelines("%g %g %g %g %g %g\n" %(sun[i],devparam[i,0], devparam[i,1],devparam[i,2],devparam[i,3],devparam[i,4]))
        f3.close()


#print(ts)        
#print(tf)
print((TimeEnd - TimeStart).total_seconds())