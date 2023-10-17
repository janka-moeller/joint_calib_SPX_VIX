"""
Created on Sat Jan 28 18:08:38 2023

@author: Guido Gazzani & Janka Moeller
"""


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import torch
from scipy.optimize import fsolve, minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


print('Torch version',torch.__version__)
print('Numpy version',np.__version__)
print('pandas version:',pd.__version__)
print('Joblib version:',pd.__version__)


# In[342]:
    
user="Guido"  # it is just a flag for storing in the corresponding directories of Guido or Janka 
#"Janka"


loss_flag="LINEAR_VEGA_DELTA"
#"SOFT_INDICATOR_VEGA_DELTA" 
#"LINEAR_VEGA_DELTA" 
#"SOFT_INDICATOR_VEGA_DELTA" 
#"LP" 
#"LINEAR" 
#"LP_V2" 
#"SOFT_INDICATOR" 

config='config9'

power=2 #even number

index_sel_maturities_spx=[0,2,4,6] #our way of indexing the maturities the present setup consists in 3-joint-calibs.
index_sel_maturities_vix=[0,1,2]

flag_hyper_both=True #flag to do the hyperparameter search
flag_do_first_optim=True  #one can do even two optimizations one after the other but it does not significally improve the calib.
lambda_coeff=0.35  # weight on the two loss functions for VIX and SPX

#for the inital sampling
vix_only=False #if lambda = 0 or lambda=1 you can put the corresponding escluded Index to False and you will calibrate only to one
spx_only=False


if vix_only and spx_only:
    raise NotImplementedError("vix_only and spx_only can not both be true!")
    
    
def sum_of_powers(x,y):
    '''
    Inputs:
    x,y : int, int 
    Returns: sum over i=0 to y of x^y
    '''
    if y<=0:
        return 1
    else:
        return x**y + sum_of_powers(x,y-1)
    
def number_of_parameters_gen(order_signature,comp_of_path):
    '''
    Inputs:
    order_signature : int
    comp_of_path : int
    
    Return:
    sp : d_n if d=comp_of_path, n=order_signature
    '''
    sp=sum_of_powers(comp_of_path,order_signature)
    return sp

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

########################################### The following part consists market data so it is not reproducible

maturities_vix=[r'14.0days',r'28.0days',r'49.0days',r'77.0days',r'105.0days',r'140.0days',r'259.0days']
maturities_spx=[r'14.0days',r'44.0days',r'58.0days',r'79.0days',r'107.0days',r'135.0days',r'170.0days',r'289.0days']
maturities_joint=[r'14.0days',r'28.0days',r'44.0days',r'49.0days',r'58.0days',r'77.0days',r'79.0days',r'105.0days',r'107.0days',r'135.0days',r'140.0days',r'170.0days',r'259.0days',r'289.0days']


day=r'/20210602'


maturities_name= "Mat_SPX"+str(index_sel_maturities_spx)+"_VIX"+str(index_sel_maturities_vix)


#change this to focus on a certain region of the smiles


moneyness_upperdev_vix=[1.2,1.2,2.3,3,3,3.2,3.5,3.8]
moneyness_lowerdev_vix=[0.1,0.1,0.2,0.2,0.2,0.2,0.2]

if vix_only:
    moneyness_upperdev_vix=[1.5,1.5,2,2,3,3,3]
    moneyness_lowerdev_vix=[0.1,0.1,0.2,0.2,0.2,0.2,0.2]

os.chdir(r'/scratch.global/ag_cu/Data/VIX/Processed Data'+day) #directory load VIX data locally

list_strikes_vix=[]
list_prices_vix=[]
list_of_maturities_vix=[]
list_bid_vix=[]
list_ask_vix=[]

list_spot_vix=[]
list_rate_vix=[]

for i,element in enumerate(maturities_vix):
    maturity_vix=element
    df_vix=pd.read_csv('ivol_data_maturity_'+maturity_vix)
    df_vix['strike']=df_vix['strike']/1000
    df_vix['open_interest']=df_vix['open_interest']/100 #not useful?
    scaling_flag_and_cut=True
    spot=df_vix['spot'][0]
    rate= df_vix['rate'][0]/100
    

    
    strikes_vix=np.array(df_vix['strike'])
    prices_vix=np.array(df_vix['mid'])
    bid_vix=np.array(df_vix['best_bid'])
    ask_vix=np.array(df_vix['best_offer'])
    
    
    
        
    idx_lowest_moneyness=find_nearest(strikes_vix,(1-moneyness_lowerdev_vix[i])*spot)
    idx_highest_moneyness=find_nearest(strikes_vix,(1+moneyness_upperdev_vix[i])*spot)
    
    strikes_vix=strikes_vix[idx_lowest_moneyness:idx_highest_moneyness+1]
    prices_vix=prices_vix[idx_lowest_moneyness:idx_highest_moneyness+1]
    bid_vix=bid_vix[idx_lowest_moneyness:idx_highest_moneyness+1]
    ask_vix=ask_vix[idx_lowest_moneyness:idx_highest_moneyness+1]    
        
    list_bid_vix.append(bid_vix)
    list_ask_vix.append(ask_vix)  
    list_rate_vix.append(rate)
        
        
    maturity=int(maturity_vix.split('.')[0])/365.25
    list_of_maturities_vix.append(maturity)
    list_strikes_vix.append(strikes_vix)
    list_prices_vix.append(prices_vix)
    list_spot_vix.append(spot)

list_of_maturities_vix=np.array(list_of_maturities_vix)
nbr_maturities_vix=len(list_of_maturities_vix)



dict_info_dataset_vix=dict(zip(range(nbr_maturities_vix),[len(list_strikes_vix[j]) for j in range(nbr_maturities_vix)]))



for j in range(nbr_maturities_vix):
    plt.plot(list_strikes_vix[j],list_prices_vix[j],marker='o',alpha=0.8,label='T={}'.format(round(list_of_maturities_vix[j],4)))
    plt.legend()
plt.title('VIX prices day '+day[-2:]+'.'+day[-4:-2]+'.'+day[1:5])
plt.show()




#guyon version
moneyness_upperdev_spx=[0.05,0.05,0.2,0.25,0.3,0.35,0.45,0.5]
moneyness_lowerdev_spx=[0.08,0.3,0.2,0.25,0.3,0.4,0.5,0.5]




os.chdir(r'/scratch.global/ag_cu/Data/SPX/Processed Data'+day) #directory load SPX data locally

list_bid_spx=[]
list_ask_spx=[]
list_strikes_spx=[]
list_prices_spx=[]
list_of_maturities_spx=[]

nbr_strikes_spx=[21,21,21,21,21,21,21,21]

for i,element in enumerate(maturities_joint):
    
    if element in maturities_spx:
        maturity_spx=element
        
        for j, mat in enumerate(maturities_spx):
            if mat==maturity_spx:
                i=np.copy(j)
        
        df_spx=pd.read_csv('ivol_data_maturity_'+maturity_spx)
        df_spx['strike']=df_spx['strike']/1000
        df_spx['open_interest']=df_spx['open_interest']/100
        spot=df_spx['spot'][0]
        
        strikes_spx=np.array(df_spx['strike'])
        prices_spx=np.array(df_spx['mid'])
        bid_spx=np.array(df_spx['best_bid'])
        ask_spx=np.array(df_spx['best_offer'])
    
        strikes_spx=(strikes_spx/spot)
        prices_spx=(prices_spx/spot)
        bid_spx=(bid_spx/spot)
        ask_spx=(ask_spx/spot)
        
        nbr_stk=len(strikes_spx)
        
        idx_lowest_moneyness=find_nearest(strikes_spx,1-moneyness_lowerdev_spx[i])
        idx_highest_moneyness=find_nearest(strikes_spx,1+moneyness_upperdev_spx[i])
        
        strikes_spx=strikes_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
        prices_spx=prices_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
        bid_spx=bid_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
        ask_spx=ask_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
        
        integers=np.array([int(b) for b in np.linspace(0,len(strikes_spx)-1,nbr_strikes_spx[i])])
        strikes_spx=strikes_spx[integers]    
        prices_spx=prices_spx[integers]
        bid_spx=bid_spx[integers]
        ask_spx=ask_spx[integers]
        
        list_bid_spx.append(bid_spx)
        list_ask_spx.append(ask_spx)
        
        
        maturity=int(maturity_spx.split('.')[0])/365.25
        print(maturity)
        list_of_maturities_spx.append(maturity)
        list_strikes_spx.append(strikes_spx)
        list_prices_spx.append(prices_spx)
        
    else: 
        list_bid_spx.append(-99)
        list_ask_spx.append(-99)
        list_strikes_spx.append(-99)
        list_prices_spx.append(-99)
        list_of_maturities_spx.append(-99)

list_of_maturities_spx=np.array(list_of_maturities_spx)
nbr_maturities_spx=len(maturities_spx)


############################################### end market data


flag_mat=True
N=int(maturity*365.25) # put here the daily sampling
T=maturity # put here maturities
#redundants



if int(config[-1])==9:  
    d=4
    order_signature=3
    sigmas=[0.7,10,5,1]
    kappas=[0.1,25,10,0]
    X0=[1,0.08,2,0]
    thetas=[0.1,4,0.08,0]
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness/n=2/config2')
    Rho =np.load('Rho_d=4.npy')



dim=d+1

print('sigmas:\n',sigmas)
print('kappas:\n',kappas)
print('thetas:\n',thetas)
print('\n')
print('Dimension expected/number of parameters to calibrate:',number_of_parameters_gen(order_signature,d))
d_star=number_of_parameters_gen(order_signature,dim)
nbr_param=number_of_parameters_gen(order_signature*2,d+1)
Rho=Rho[:len(X0),:len(X0)]
print('Correlation matrix:\n',np.matrix(Rho).round(4))


############################################################### Here we load the pre-computed object and concatenate them

# choose configuration ##########################

os.chdir(r'/scratch.global/ag_cu/Codes_'+user+'/Randomness/n='+str(order_signature)+'/'+config)

if config[0]=="r":  ############################################## in this little chunck one can upload a randomized version of the 
    Q_0_all_truncated= np.load("rand_Q_0_all_truncated.npy")     # sampled quantities with the sampler (see comments in the paper)
    e_B_sig=np.load("rand_e_B_sig.npy")
    L_torch=np.load("rand_L_torch.npy")
    
    Q_0_all_truncated=torch.tensor(Q_0_all_truncated)
    e_B_sig= torch.tensor(e_B_sig)
    L_torch= torch.tensor(L_torch)
    
    param_dim=Q_0_all_truncated.shape[-1]

else:

    files=[]
    for element in os.listdir():
        files.append(element)
    
    
    param_dim=number_of_parameters_gen(order_signature,d)
    MC_nbr=80000
    
    Q_0_all_truncated=[]
    e_B_sig=[]
    L=[]
    
    for file in files:
        auxiliary=np.load(file,allow_pickle=True) 
        if file[0]=='E':
            e_B_sig.append(auxiliary)
        if file[0]=='L':
            L.append(auxiliary)
        if file[0]=='Q':
            Q_0_all_truncated.append(auxiliary)
            
            
    def concatenate_and_torchify(obj):
        for j in range(7): # 7 because the rounds of element save are 8; change it accordingly if one changes the nbr of rounds in the sampling procedure
            if j==0:
                aux=np.concatenate((obj[j],obj[j+1]),axis=1)
            else:
                aux=np.concatenate((aux,obj[j+1]),axis=1)
        return torch.tensor(aux)
    
    Q_0_all_truncated=concatenate_and_torchify(Q_0_all_truncated)
    e_B_sig=concatenate_and_torchify(e_B_sig)
    L_torch=concatenate_and_torchify(L)



print('Q_0_all_truncated shape',Q_0_all_truncated.shape)
print('e_B_sig shape', e_B_sig.shape)
print('L_torch shape', L_torch.shape)

    
def aux_torch(K,index_sel_maturities,norm,r,q,flag_option_type,mat):
    
    '''
    Input: K (float); strike
          index_sel_maturities, list (of indices corresponding to the desired maturities, e.g. [0] for the first mat)
          norm: VIX_T or S_T; torch.tensors (outputs of VIX_T and get_S_T functions)
    Outuput: matrix of payoffs (for one strike and all samples)
    
    '''
    if flag_option_type=='VIX':
        matrix_=[np.exp(-r*mat)*torch.maximum(norm[j,:] - K,torch.tensor(0)) for j in range(len(index_sel_maturities))]
        matrix=torch.stack(matrix_)
        return matrix
    elif flag_option_type=='SPX':

        matrix_=[np.exp(-r*mat)*torch.maximum(np.exp((r-q)*mat)*norm[j,:] - K,torch.tensor(0)) for j in range(len(index_sel_maturities))]
        matrix=torch.stack(matrix_)
        return matrix


def VIX_T(l,L_torch,annualization):
    '''
    Input: l, torch.tensor; parameters
          L_torch, torch.tensor; loaded previously
          annualization; float, scaling factor 
    '''
    scalar_product=torch.matmul(L_torch,l)**2
    res=torch.sum(scalar_product,-1)
    norm=torch.sqrt(annualization*res)
    
    norm=norm.unsqueeze(0)
    return norm


def monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities,r,mat,q=0): #Cholesky decomposition here is used to speed this up
    '''
    Pricing function for the VIX, it runs for all strikes the payoff evaluation via aux_torch
    '''
   
    mc_payoff_arr=[aux_torch(K,index_sel_maturities,VIX,r,0,'VIX',mat) for K in strikes_vix]
    return torch.stack(mc_payoff_arr).squeeze(1)



def optimal_controls(S_T,strike, rate=0, mat=0):
    '''
    Input: S_T or VIX_T; torch.tensor
           strike; float
    Output: Optimal control variate
    '''
    zero=torch.tensor(0)
    Y_cv=np.exp(rate*mat)*S_T-strike
    CV=Y_cv-torch.mean(Y_cv)
    optimal_gamma=np.cov(Y_cv.numpy(),torch.maximum(np.exp(rate*mat)*S_T-strike,zero).numpy())[0,1]
    optimal_gamma=optimal_gamma/np.var(Y_cv.numpy())
    return optimal_gamma*CV 



def controlled_price(monte_carlo):  #modify this and the function above in case to include control variates
    ''' 
    Controlled price via MC-CV
    '''
    diff=monte_carlo
    diff=torch.mean(diff,1)
    return diff.numpy()



ell=torch.tensor(np.random.uniform(-0.5,0.5,param_dim))


Delta=1/12 #approximately 30 days
annualization=(100**2)/Delta #as in CBOE


def get_S_T(ell,Q_0_all_truncated,e_B_sig):
    '''
    Inputs:
    ell: torch_tensor, (d+1)_n dimension
    Q0_all_truncated: torch.tensor, Q_0 matrix without the redundant components
    e_B_sig: torch.tensor, signature at maturity without the redundant components
    
    Returns: 
    S_t:  torch.tensor, price at maturity
    
    '''
    x=torch.matmul(Q_0_all_truncated,ell)
    quadratic_=-1/2*torch.matmul(x,ell)
    linear_=torch.matmul(e_B_sig,ell)
    log_S_t=quadratic_+linear_
    S_t=torch.exp(log_S_t)
    return S_t

def monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities,r,q,mat): #Cholesky decomposition here is NOT used 
    '''
    Pricing function for the SPX, it runs for all strikes the payoff evaluation via aux_torch
    '''

    mc_payoff_arr=[aux_torch(K,index_sel_maturities,S_T,r,q,'SPX',mat) for K in strikes_spx]
    return torch.stack(mc_payoff_arr).squeeze(1)


# Auxiliary function Implied Volatility and Las Vegas :)


def phi(x): ## Gaussian density
    return np.exp(-x*x/2.)/np.sqrt(2*np.pi)   

def find_ivol(price, spot, strike, T, r, d):

    def BS_price(sigma):
        d_1= 1/(sigma*np.sqrt(T))*(np.log(spot/strike) + (r-d+sigma**2/2)*T)
        d_2= d_1-sigma*np.sqrt(T)
        
        N_1= norm.cdf(d_1) #scipy.stats.norm.cdf
        N_2= norm.cdf(d_2) #scipy.stats.norm.cdf
        
        #print(N_1, flush=True)
        return N_1*spot*np.exp(-d*T)-N_2*strike*np.exp(-r*T) - price
    
    root = fsolve(BS_price, 1)[-1] #scipy.optimize.fsolve
    
    if np.isclose(BS_price(root), 0.0):
        return root
    
    else:
        return -99.99

def Las_Vegas(spot,strike,T,r,d,sigma):
    d_1= 1/(sigma*np.sqrt(T))*(np.log(spot/strike) + (r-d+sigma**2/2)*T)
    return np.exp(-d*T)*spot*np.sqrt(T)*phi(d_1)
 
def Las_Deltas(spot,strike,T,r,d,sigma):
    d_1= 1/(sigma*np.sqrt(T))*(np.log(spot/strike) + (r-d+sigma**2/2)*T)
    N_1= norm.cdf(d_1) #scipy.stats.norm.cdf
    return N_1*np.exp(-d*T) 

os.chdir(r'/scratch.global/ag_cu/Data/SPX/Processed Data'+day)

iv_surface_spx=[]
las_vegas_spx=[]
threshold_spx=100  
threshold_vix=25
bid_ask_surface_spx=[]

list_rate_spx=[]
list_divi_spx=[]

bid_ivol_spx=[]
ask_ivol_spx=[]

for j,element in enumerate(maturities_joint):
    
    if element in maturities_spx:
        maturity_spx_aux=element
        df_spx=pd.read_csv('ivol_data_maturity_'+maturity_spx_aux)
        rate_spx=df_spx['rate'][0]/100
        divi_spx=df_spx['divi'][0]/100
        list_rate_spx.append(rate_spx)
        list_divi_spx.append(divi_spx)
        
        
        maturity_spx=list_of_maturities_spx[j]
        strikes_spx=list_strikes_spx[j]
        prices_spx=list_prices_spx[j]
        bid_spx=list_bid_spx[j]
        ask_spx=list_ask_spx[j]
        
        
        iv_per_mat=[find_ivol(prices_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]
        
        bid_per_mat=[find_ivol(bid_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]
        
        ask_per_mat=[find_ivol(ask_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]
        
        las_vegas=[min(1/Las_Vegas(1,strike,maturity_spx,rate_spx,divi_spx,iv_per_mat[k]),threshold_spx) for k,strike in enumerate(strikes_spx)]
        las_vegas=np.array(las_vegas)
        
        plt.plot(strikes_spx,np.array(iv_per_mat),color='blue')
        plt.scatter(strikes_spx,np.array(bid_per_mat),marker='+',color='red',label='Bid')
        plt.scatter(strikes_spx,np.array(ask_per_mat),marker='+',color='green',label='Ask')
        plt.legend()
        plt.title('T= '+element) ####################### ADD HERE SAVE FIG to save the smiles and bid-ask for SPX
        plt.show()
        
        plt.plot(strikes_spx,las_vegas,color='blue',marker='o',alpha=0.7)
        plt.title('Vega weights T= '+element)
        plt.show() 
        
        las_vegas_spx.append(las_vegas)
        iv_surface_spx.append(np.array(iv_per_mat))
        bid_ask_surface_spx.append(np.array([bid_per_mat,ask_per_mat]))
        
        bid_ivol_spx.append(np.array(bid_per_mat))
        ask_ivol_spx.append(np.array(ask_per_mat))
    else:
        iv_surface_spx.append(-99)
        las_vegas_spx.append(-99)

        bid_ask_surface_spx.append(-99)

        list_rate_spx.append(-99)
        list_divi_spx.append(-99)
        
        bid_ivol_spx.append(-99)
        ask_ivol_spx.append(-99)
    
list_rate_spx=np.array(list_rate_spx)
list_divi_spx=np.array(list_divi_spx)

os.chdir(r'/scratch.global/ag_cu/Data/VIX/Processed Data'+day)

iv_surface_vix=[]
las_vegas_vix=[]
las_deltas_vix=[] 
bid_ask_surface_vix=[]

ask_ivol_vix=[]
bid_ivol_vix=[]


for j,element in enumerate(maturities_vix):
    maturity_vix_aux=element
    df_vix=pd.read_csv('ivol_data_maturity_'+maturity_vix_aux)
    rate_vix=df_vix['rate'][0]/100
    spot_vix=df_vix['spot'][0]
    #divi_vix=df_vix['divi'][0]/100
    
    
    maturity_vix=list_of_maturities_vix[j]
    strikes_vix=list_strikes_vix[j]
    prices_vix=list_prices_vix[j]
    bid_vix=list_bid_vix[j]
    ask_vix=list_ask_vix[j]
    
    
    iv_per_mat=[find_ivol(prices_vix[k], np.exp(-rate_vix*maturity_vix)*spot_vix, strike, maturity_vix, rate_vix, 0) for k,strike in enumerate(strikes_vix)]
    
    bid_per_mat=[find_ivol(bid_vix[k], np.exp(-rate_vix*maturity_vix)*spot_vix, strike, maturity_vix, rate_vix, 0) for k,strike in enumerate(strikes_vix)]
    
    ask_per_mat=[find_ivol(ask_vix[k], np.exp(-rate_vix*maturity_vix)*spot_vix, strike, maturity_vix, rate_vix, 0) for k,strike in enumerate(strikes_vix)]
    
    las_vegas=[min(1/Las_Vegas(np.exp(-rate_vix*maturity_vix)*spot_vix,strike,maturity_vix,rate_vix,0,iv_per_mat[k]),threshold_vix) for k,strike in enumerate(strikes_vix)]
    las_vegas=np.array(las_vegas)
    las_deltas=[Las_Deltas(np.exp(-rate_vix*maturity_vix)*spot_vix,strike,maturity_vix,rate_vix,0,iv_per_mat[k]) for k,strike in enumerate(strikes_vix)]
    las_deltas=np.array(las_deltas)


    #las_vegas= las_vegas/np.mean(las_vegas)
    print("las_vegas VIX:", las_vegas)

    plt.figure()
    plt.plot(strikes_vix,np.array(iv_per_mat),color='blue')
    plt.scatter(strikes_vix,np.array(bid_per_mat),marker='+',color='red',label='Bid')
    plt.scatter(strikes_vix,np.array(ask_per_mat),marker='+',color='green',label='Ask')
    plt.legend()
    plt.title('T= '+element) ####################### ADD HERE SAVE FIG to save the smiles and bid-ask for VIX
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,las_vegas,color='blue',marker='o',alpha=0.7)
    plt.title('Vegas weights T= '+element) ####################### ADD HERE SAVE FIG to save Las Vegas weights for VIX
    plt.show()
    
    
    las_vegas_vix.append(las_vegas)
    las_deltas_vix.append(las_deltas) 
    iv_surface_vix.append(np.array(iv_per_mat))
    bid_ask_surface_vix.append(np.array([bid_per_mat,ask_per_mat]))

    ask_ivol_vix.append(np.array(ask_per_mat))
    bid_ivol_vix.append(np.array(bid_per_mat))




weights_spx=np.array([[1 for _ in range(len(las_vegas_spx[idx]))] for idx in index_sel_maturities_spx])
weights_vix=np.array([[1 for _ in range(len(las_vegas_vix[idx]))] for idx in index_sel_maturities_vix])



    
def quadratic_form(l,Q):
    '''
    Inputs: 
    l: torch.tensor, parameters to be calibrated
    Q: torch.tensor, dimension MC_number x dim(l) x dim(l)
    Returns: l^T Q l, i.e., polynomial in l: torch.tensor
    '''
    polynomial_l=torch.matmul(l,Q)
    polynomial_l=torch.matmul(polynomial_l,l)
    return polynomial_l




if flag_hyper_both==True:
    
    nbr_samples_ell=10000
    
    Q_0_all_truncated_only_rel= Q_0_all_truncated[:(index_sel_maturities_spx[-1]+1), :, :, :]
    e_B_sig_only_rel= e_B_sig[:(index_sel_maturities_spx[-1]+1), :, :]
    L_torch_only_rel= L_torch[:index_sel_maturities_vix[-1], :, :, :]
    
    
    ell_sampled=[]
    error_spx = []
    error_spx_1 = []
    error_vix = []
    error_joint=[]
    sampled_prices_spx= []
    sampled_prices_spx_1= []
    sampled_prices_vix= []
    
    sampled_vix_futures=[]
    
    bid_ask_spread_vix= np.array(list_bid_vix)-np.array(list_ask_vix)
    bid_ask_spread_spx= np.array(list_bid_spx)-np.array(list_ask_spx)
    
    
    n_param= Q_0_all_truncated_only_rel.shape[-1]
   
    for j in range(nbr_samples_ell):
        ell=torch.tensor(np.random.uniform(-0.1,0.1,Q_0_all_truncated_only_rel.shape[-1]))
        ell_sampled.append(ell)
        

        spx_model_list=[]
        for idx in index_sel_maturities_spx:
            S_T=get_S_T(ell,Q_0_all_truncated_only_rel[idx,:,:,:],e_B_sig_only_rel[idx,:,:])
            S_T=S_T.unsqueeze(0)
            monte_carlo=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
            spx_model_prices=controlled_price(monte_carlo) 
            spx_model_list.append(spx_model_prices)
        
        
        vix_model_list=[]
        vix_futures_list=[]
        VIX_total=VIX_T(ell,L_torch,annualization)
        for idx in index_sel_maturities_vix:
            VIX=VIX_total[:,idx,:]
            
            monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
            vix_model_prices=controlled_price(monte_carlo_vix)
            vix_model_list.append(vix_model_prices)
            
            VIX_future= torch.mean(VIX)
            vix_futures_list.append(VIX_future)
            
        sampled_prices_spx.append(spx_model_list)
        sampled_prices_vix.append(vix_model_list)
        
        sampled_vix_futures.append(vix_futures_list)
        
        err_spx=0
        for i,idx in enumerate(index_sel_maturities_spx):
            err_spx+=np.sum(np.abs((list_prices_spx[idx]-spx_model_list[i])/bid_ask_spread_spx[idx])) 
        error_spx.append(err_spx) 
        
        
        err_vix=0
        for i,idx in enumerate(index_sel_maturities_vix):
            err_vix+=np.sum(np.abs((list_prices_vix[idx]-vix_model_list[i])/bid_ask_spread_vix[idx]))
            err_vix+=np.abs(list_spot_vix[idx]-vix_futures_list[i])/list_spot_vix[idx]
            
        error_vix.append(err_vix) 
        
        error_joint.append(err_vix+err_spx)
        
        

    error_spx=np.array(error_spx)
    error_vix=np.array(error_vix)
    error_joint=np.array(error_joint)

    
    if vix_only:
        idx_min =np.where( error_vix==np.min( error_vix))[0][0]
    elif spx_only:
        idx_min =np.where( error_spx==np.min(error_spx))[0][0]
    else: 
        idx_min =np.where( error_joint==np.min( error_joint))[0][0]
    
    
    
    print('Minimal error on BOTH MC:', error_joint[idx_min])
    print('Minimal error on BOTH for SPX:', error_spx[idx_min])
    print('Minimal error on BOTH for VIX:', error_vix[idx_min])
    

    ell_init=ell_sampled[idx_min]
    init_spx_model_prices=sampled_prices_spx[idx_min]
    init_vix_model_prices=sampled_prices_vix[idx_min]

    
    if vix_only:
        save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/ALL_JOINT_same_ell/vix_only/'+maturities_name+r'/'+config
    elif spx_only:
        save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/ALL_JOINT_same_ell/spx_only/'+maturities_name+r'/'+config
    else:
        save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/ALL_JOINT_same_ell/'+maturities_name+r'/'+config

    os.makedirs(save_init_dir, exist_ok=True)
    os.chdir(save_init_dir)
    
    for i, idx in enumerate(index_sel_maturities_spx):
        plt.figure()
        plt.plot(list_strikes_spx[idx],init_spx_model_prices[i],marker='o',label='Model')
        plt.plot(list_strikes_spx[idx],list_prices_spx[idx],marker='*',label='Market')
        plt.xlabel('Strikes')
        plt.legend()
        plt.title('SPX'+str(i))
        plt.savefig("SPX"+str(i))
        plt.show()
    
    
    
    for i, idx in enumerate(index_sel_maturities_spx):
        plt.figure()
        plt.plot(list_strikes_spx[idx],np.abs(init_spx_model_prices[i]-list_prices_spx[idx]),marker='o')
        plt.xlabel('Strikes')
        plt.title('Absolute Error SPX'+str(i))
        plt.savefig("Error SPX"+str(i))
        plt.show()
    
    
    for i, idx in enumerate(index_sel_maturities_vix):
        plt.figure()
        plt.plot(list_strikes_vix[idx],init_vix_model_prices[i],marker='o',label='Model')
        plt.plot(list_strikes_vix[idx],list_prices_vix[idx],marker='*',label='Market')
        plt.axvline(x=list_spot_vix[idx], ls="--", label="market future", color="red")
        plt.axvline(x=sampled_vix_futures[idx_min][i], ls="--", label="model future", color="blue")
        plt.xlabel('Strikes')
        plt.legend()
        plt.title('VIX'+str(i))
        plt.savefig("VIX"+str(i))
        plt.show()
    

    for i, idx in enumerate(index_sel_maturities_vix):
        plt.figure()
        plt.plot(list_strikes_vix[idx],np.abs(init_vix_model_prices[i]-list_prices_vix[idx]),marker='o')
        plt.xlabel('Strikes')
        plt.title('Absolute Error VIX'+str(i))
        plt.savefig("Error VIX"+str(i))
        plt.show()

    
    np.save('ell_init_both.npy',ell_init)



else:
    if vix_only:
        save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/ALL_JOINT_same_ell/vix_only/'+maturities_name+r'/'+config
    elif spx_only:
        save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/ALL_JOINT_same_ell/spx_only/'+maturities_name+r'/'+config
    else:
        save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/ALL_JOINT_same_ell/'+maturities_name+r'/'+config

    os.chdir(save_init_dir)
    
    ell_init= np.load('ell_init_both.npy')
    
    ell_init= torch.tensor(ell_init)
    

def soft_indicator(x):  #### to force us to be in the bid-ask spread
    return 0.5*(np.tanh(x*100)+1)


def loss_vix_soft_indicator(l):
    l=torch.tensor(l)
    diff=0
    for i,idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        controlled_vix=controlled_price(monte_carlo_vix) 
        VIX_future= torch.mean(VIX)
        diff+=np.mean(np.multiply((list_prices_vix[idx]-controlled_vix)**2,((1/np.abs(list_bid_vix[idx]-list_ask_vix[idx]))*las_vegas_vix[i])**2)*(soft_indicator(list_bid_vix[idx]-controlled_vix)+soft_indicator(controlled_vix-list_ask_vix[idx])))
        diff+=10*(VIX_future.numpy()-list_spot_vix[idx])**2
    return diff

def loss_vix_soft_indicator_vega(l):
    l=torch.tensor(l)
    diff=0
   
    for i,idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        controlled_vix=controlled_price(monte_carlo_vix) 
        VIX_future= torch.mean(VIX)
        diff+=np.mean(np.multiply((list_prices_vix[idx]-controlled_vix)**2,((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx])**2)*(soft_indicator(list_bid_vix[idx]-controlled_vix)+soft_indicator(controlled_vix-list_ask_vix[idx])))
        diff+=10*(VIX_future.numpy()-list_spot_vix[idx])**2
    return diff

def loss_vix_linear(l):
    l=torch.tensor(l)
    diff=0
    for i,idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        controlled_vix=controlled_price(monte_carlo_vix)
        VIX_future= torch.mean(VIX)
        diff+=np.mean(np.multiply((list_prices_vix[idx]-controlled_vix)**2,((1/np.abs(list_bid_vix[idx]-list_ask_vix[idx]))*weights_vix[i])**2))
        diff+=10*(VIX_future.numpy()-list_spot_vix[idx])**2
    return diff

def loss_vix_linear_vega_delta(l):
    l=torch.tensor(l)
    diff=0
    for i,idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        controlled_vix=controlled_price(monte_carlo_vix)
        
        VIX_future= torch.mean(VIX)
        
        vix_future_arr= np.array([VIX_future.numpy()]*len(list_strikes_vix[idx]))
        vix_spot_arr= np.array([list_spot_vix[idx]]*len(list_strikes_vix[idx]))

    
        diff_1= np.abs(np.multiply((list_prices_vix[idx]-controlled_vix),((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx])))
    

        diff_2= np.abs(np.multiply((vix_spot_arr-vix_future_arr),((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx]*las_deltas_vix[idx])))

        diff+= np.mean((diff_1+diff_2)**power)
    return diff

def loss_vix_softindicator_vega_delta(l):
    l=torch.tensor(l)
    diff=0
    for i,idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
      
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        controlled_vix=controlled_price(monte_carlo_vix)
        VIX_future= torch.mean(VIX) 
        vix_future_arr= np.array([VIX_future.numpy()]*len(list_strikes_vix[idx]))
        vix_spot_arr= np.array([list_spot_vix[idx]]*len(list_strikes_vix[idx]))
   
    
        diff_1= np.abs(np.multiply((list_prices_vix[idx]-controlled_vix),((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx])))*(soft_indicator(list_bid_vix[idx]-controlled_vix)+soft_indicator(controlled_vix-list_ask_vix[idx]))

        diff_2= np.abs(np.multiply((vix_spot_arr-vix_future_arr),((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx]*las_deltas_vix[idx])))

        diff+= np.mean((diff_1+diff_2)**power)
    return diff

def loss_spx_soft_indicator(l): 
    l=torch.tensor(l)
    diff=0
    for i, idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:]).unsqueeze(0)
        monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        controlled_spx=controlled_price(monte_carlo_spx) #optimal_gammas_spx,
        diff+=np.mean(np.multiply((list_prices_spx[idx]-controlled_spx)**2,((1/np.abs(list_bid_spx[idx]-list_ask_spx[idx]))*weights_spx[i])**2)*(soft_indicator(list_bid_spx[idx]-controlled_spx)+soft_indicator(controlled_spx-list_ask_spx[idx])))
    return diff


def loss_spx_soft_indicator_vega(l): 
    l=torch.tensor(l)
    diff=0
    for i, idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:]).unsqueeze(0)
        monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        controlled_spx=controlled_price(monte_carlo_spx)
        diff+=np.mean(np.multiply((list_prices_spx[idx]-controlled_spx)**2,((1/np.abs(bid_ivol_spx[idx]-ask_ivol_spx[idx]))*las_vegas_spx[idx])**2)*(soft_indicator(list_bid_spx[idx]-controlled_spx)+soft_indicator(controlled_spx-list_ask_spx[idx])))
    return diff

def loss_spx_linear(l):
    l=torch.tensor(l)
    diff=0
    for i, idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:]).unsqueeze(0)
        monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        controlled_spx=controlled_price(monte_carlo_spx) #optimal_gammas_spx,

        diff+=np.mean(np.multiply((list_prices_spx[idx]-controlled_spx)**2,((1/np.abs(list_bid_spx[idx]-list_ask_spx[idx]))*weights_spx[i])**2))
    return diff

def loss_spx_linear_vega(l):
    l=torch.tensor(l)
    diff=0
    for i, idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:]).unsqueeze(0)
        monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        controlled_spx=controlled_price(monte_carlo_spx) #optimal_gammas_spx,
        diff+=np.mean(np.multiply((list_prices_spx[idx]-controlled_spx)**power,((1/np.abs(bid_ivol_spx[idx]-ask_ivol_spx[idx]))*las_vegas_spx[idx])**power))
    return diff


def loss_spx_softindicator_vega(l): 
    l=torch.tensor(l)
    diff=0
    for i, idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:]).unsqueeze(0)
        monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        controlled_spx=controlled_price(monte_carlo_spx)
        diff+=np.mean(np.multiply((list_prices_spx[idx]-controlled_spx)**power,((1/np.abs(bid_ivol_spx[idx]-ask_ivol_spx[idx]))*las_vegas_spx[idx])**power)*(soft_indicator(list_bid_spx[idx]-controlled_spx)+soft_indicator(controlled_spx-list_ask_spx[idx])))
    return diff

comp_price_scale=1


def loss_joint_soft_indicator(l):
    diff1=loss_spx_soft_indicator(l)
    diff2=loss_vix_soft_indicator(l)
    res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
    return res 

def loss_joint_soft_indicator_vega(l):
    diff1=loss_spx_soft_indicator_vega(l)
    diff2=loss_vix_soft_indicator_vega(l)
    res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
    return res 

def loss_joint_linear(l):
    diff1=loss_spx_linear(l)
    diff2=loss_vix_linear(l)
    res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
    return res 


def loss_joint_linear_vega_delta(l):
    diff1=loss_spx_linear_vega(l)
    diff2=loss_vix_linear_vega_delta(l)
    res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
    return res 

def loss_joint_softindicator_vega_delta(l):
    diff1=loss_spx_softindicator_vega(l)
    diff2=loss_vix_softindicator_vega_delta(l)
    res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
    return res 

exp=50
def rho(x):
    function=(1+x)**exp
    first_derivative= exp*(1+x)**(exp-1)
    second_derivative= exp*(exp-1)*(1+x)**(exp-2)
    return np.array([function, first_derivative, second_derivative])

def rho_v2(x):
    function=(x)**exp
    first_derivative= exp*(x)**(exp-1)
    second_derivative= exp*(exp-1)*(x)**(exp-2)
    return np.array([function, first_derivative, second_derivative])

 

if flag_do_first_optim: 
    for h in tqdm(range(1),desc='Calibration 1'):

        if loss_flag=="SOFT_INDICATOR":
            res= minimize(loss_joint_soft_indicator, ell_init.numpy())#, options={'maxiter':10})
        elif loss_flag=="SOFT_INDICATOR_VEGA":
            res= minimize(loss_joint_soft_indicator_vega, ell_init.numpy())#, options={'maxiter':10})
        elif loss_flag=="SOFT_INDICATOR_VEGA_DELTA":
            res= minimize(loss_joint_softindicator_vega_delta, ell_init.numpy())#, options={'maxiter':10})
        elif loss_flag=="LINEAR_VEGA_DELTA":
            res= minimize(loss_joint_linear_vega_delta, ell_init.numpy())#, options={'maxiter':10})
        elif loss_flag=="LINEAR":
            res= minimize(loss_joint_linear, ell_init.numpy())#, options={'maxiter':10})

 
            
    if vix_only:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/vix_only/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
    elif spx_only:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/spx_only/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
    else:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)

    os.makedirs(save_dir, exist_ok=True)
    
    os.chdir(save_dir)
    

    np.save('ell_optimal_first_joint.npy',res['x'])
    
    l=torch.tensor(res['x'])
    
    spx_model_list=[]
    for i,idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:])
        S_T=S_T.unsqueeze(0)
        monte_carlo=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        spx_model_prices=controlled_price(monte_carlo) #optimal_gammas,
        spx_model_list.append(spx_model_prices)

        np.save('prices_optimal_SPX_'+str(i)+'.npy', spx_model_prices)
        
        plt.figure()
        plt.plot(list_strikes_spx[idx],spx_model_prices,marker='o',label='Model')
        plt.plot(list_strikes_spx[idx],list_prices_spx[idx],marker='*',label='Market')
        plt.xlabel('Strikes')
        plt.legend()
        plt.title('SPX'+str(i))
        plt.savefig("Calibrated_SPX_"+str(i))
        plt.show()
        
        plt.figure()
        plt.plot(list_strikes_spx[idx],np.abs(spx_model_prices-list_prices_spx[idx]),marker='o')
        plt.xlabel('Strikes')
        plt.title('Absolute Error SPX'+str(i))
        plt.savefig("Calibrated Error_SPX_"+str(i))
        plt.show()
        
    
   
    vix_model_list=[]
    vix_futures_list=[]
    VIX_total=VIX_T(l,L_torch,annualization)
    for i, idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_total[:,idx,:]
        VIX_future= torch.mean(VIX)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        vix_model_prices=controlled_price(monte_carlo_vix) 
        vix_model_list.append(vix_model_prices)
        vix_futures_list.append(VIX_future) 
        
        print("VIX_future "+str(i)+" without rate:", VIX_future)
        print("VIX_future "+str(i)+" with rate:", VIX_future*torch.tensor(np.exp(-1*list_rate_vix[idx]*list_of_maturities_vix[idx])))
        print("VIX spot"+str(i)+":", list_spot_vix[idx])
        
        np.save('prices_optimal_VIX_'+str(i)+'.npy',vix_model_prices)
        
        plt.figure()
        plt.plot(list_strikes_vix[idx],vix_model_prices,marker='o',label='Model')
        plt.plot(list_strikes_vix[idx],list_prices_vix[idx],marker='*',label='Market')
        plt.axvline(x=list_spot_vix[idx], ls="--", label="market future", color="red")
        plt.axvline(x=VIX_future, ls="--", label="model future", color="blue")

        plt.xlabel('Strikes')
        plt.legend()
        plt.title('VIX'+str(i))
        plt.savefig("Calibrated_VIX_"+str(i))
        plt.show()
        
        plt.figure()
        plt.plot(list_strikes_vix[idx],np.abs(vix_model_prices-list_prices_vix[idx]),marker='o')
        plt.xlabel('Strikes')
        plt.title('Absolute Error VIX'+str(i))
        plt.savefig("Calibrated Error_VIX_"+str(i))
        plt.show()
        
    np.save('optimal_VIX_futures.npy',vix_futures_list)    
    
else: 
    
    if vix_only:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/vix_only/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
    elif spx_only:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/spx_only/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
    else:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)

    
    os.chdir(save_dir)
    
    
    l=np.load('ell_optimal_first_joint.npy')
    l= torch.tensor(l)
    
    spx_model_list=[]
    for i,idx in enumerate(index_sel_maturities_spx):
        S_T=get_S_T(l,Q_0_all_truncated[idx,:,:,:],e_B_sig[idx,:,:])
        S_T=S_T.unsqueeze(0)
        monte_carlo=monte_carlo_pricing_spx_nochol(S_T,list_strikes_spx[idx],[idx],list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        spx_model_prices=controlled_price(monte_carlo) #optimal_gammas,
        spx_model_list.append(spx_model_prices)

        np.save('prices_optimal_SPX_'+str(i)+'.npy', spx_model_prices)
        
        plt.figure()
        plt.plot(list_strikes_spx[idx],spx_model_prices,marker='o',label='Model')
        plt.plot(list_strikes_spx[idx],list_prices_spx[idx],marker='*',label='Market')
        plt.xlabel('Strikes')
        plt.legend()
        plt.title('SPX'+str(i))
        plt.savefig("Calibrated_SPX_"+str(i))
        plt.show()
        
        plt.figure()
        plt.plot(list_strikes_spx[idx],np.abs(spx_model_prices-list_prices_spx[idx]),marker='o')
        plt.xlabel('Strikes')
        plt.title('Absolute Error SPX'+str(i))
        plt.savefig("Calibrated Error_SPX_"+str(i))
        plt.show()
        
    
   
    vix_model_list=[]
    vix_futures_list=[]
    VIX_total=VIX_T(l,L_torch,annualization)
    for i, idx in enumerate(index_sel_maturities_vix):
        VIX=VIX_total[:,idx,:]
        VIX_future= torch.mean(VIX)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,list_strikes_vix[idx],[idx],list_rate_vix[idx],list_of_maturities_vix[idx])
        vix_model_prices=controlled_price(monte_carlo_vix) 
        vix_model_list.append(vix_model_prices)
        vix_futures_list.append(VIX_future)
        
        print("VIX_future "+str(i)+" without rate:", VIX_future)
        print("VIX_future "+str(i)+" with rate:", VIX_future*torch.tensor(np.exp(-1*list_rate_vix[idx]*list_of_maturities_vix[idx])))
        print("VIX spot"+str(i)+":", list_spot_vix[idx])
        
        
        np.save('prices_optimal_VIX_'+str(i)+'.npy',vix_model_prices)
        
        plt.figure()
        plt.plot(list_strikes_vix[idx],vix_model_prices,marker='o',label='Model')
        plt.plot(list_strikes_vix[idx],list_prices_vix[idx],marker='*',label='Market')
        plt.axvline(x=list_spot_vix[idx], ls="--", label="market future", color="red")
        plt.axvline(x=VIX_future, ls="--", label="model future", color="blue")

        plt.xlabel('Strikes')
        plt.legend()
        plt.title('VIX'+str(i))
        plt.savefig("Calibrated_VIX_"+str(i))
        plt.show()
        
        plt.figure()
        plt.plot(list_strikes_vix[idx],np.abs(vix_model_prices-list_prices_vix[idx]),marker='o')
        plt.xlabel('Strikes')
        plt.title('Absolute Error VIX'+str(i))
        plt.savefig("Calibrated Error_VIX_"+str(i))
        plt.show()
        
    np.save('optimal_VIX_futures.npy',vix_futures_list)  

  