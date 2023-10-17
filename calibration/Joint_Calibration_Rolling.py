"""
@author: Guido Gazzani & Janka Moeller
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import torch
from joblib import Parallel, delayed ########################## It would be nice to use it somewhere to speed up the calibration but I tried and it 
############################################################### actually slows it down.
from scipy.optimize import least_squares, fsolve, minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import random_correlation, uniform
#import signatory 

#print('Signatory version:',signatory.__version__)
print('Torch version',torch.__version__)
print('Numpy version',np.__version__)
print('pandas version:',pd.__version__)
print('Joblib version:',pd.__version__)


# In[342]:
    
user="Janka"
config="config8"


loss_flag="LINEAR_VEGA_DELTA" #"LP" # "SOFT_INDICATOR" "LINEAR" "LP_V2"

global power
power=2

lambda_coeff=1

load_dir_1= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/Maturity_1_SPX/'+config

load_dir_2= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/Maturity_2_SPX_1_VIX/'+config+'/unscaled_vix/cutt_spx/corrected_version/'+loss_flag+'_lambda='+str(lambda_coeff)

# load_precomputed_vixmat=[r'14.0days', r'49.0days']
# load_precomputed_dir= [load_dir_1, load_dir_2]
# load_precomputed_file= ["ell_optimal.npy","ell_optimal_first_joint_si.npy"]

# index_sel_maturities_spx=[6]
# index_sel_maturities_vix=[2]



# load_precomputed_vixmat=[r'14.0days']
# load_precomputed_dir= [load_dir_1]
# load_precomputed_file= ["ell_optimal.npy"]


# index_sel_maturities_spx=[2]
# index_sel_maturities_vix=[0]

###############################################################

load_dir_joint1= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/Mat_SPX[0, 2]_VIX[0]/'+config+'/LINEAR_VEGA_DELTA2_lambda=0.35'
load_dir_joint2= load_dir_joint1+ r"/ROLLING/Maturity_SPX[6]_VIX[2]/"+config+"/LINEAR_VEGA_DELTA2_lambda=0.35/took_last"
load_dir_joint3= load_dir_joint2+ r"/ROLLING/Maturity_SPX[9]_VIX[4]/"+config+"/LINEAR_VEGA_DELTA2_lambda=0.35/took_last"
load_dir_joint4= load_dir_joint3+ r"/ROLLING/Maturity_SPX[11]_VIX[5]/"+config+"/LINEAR_VEGA_DELTA2_lambda=0.35/took_last"


load_precomputed_vixmat= [r'49.0days', r'105.days', r'140.days']#[r'49.0days', r'105.days', r'140.days', r'259.0days'] 
load_precomputed_dir= [load_dir_joint1, load_dir_joint2, load_dir_joint3] #[load_dir_joint1, load_dir_joint2, load_dir_joint3, load_dir_joint4]
load_precomputed_file= ["ell_optimal_first_joint.npy","ell_optimal_first_joint.npy", "ell_optimal_first_joint.npy"] #["ell_optimal_first_joint.npy","ell_optimal_first_joint.npy","ell_optimal_first_joint.npy","ell_optimal_first_joint.npy"]

index_sel_maturities_spx=[11]
index_sel_maturities_vix=[5]

# load_dir_joint1= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/Mat_SPX[0, 2, 4]_VIX[0, 1]/config9/SOFT_INDICATOR_VEGA_DELTA2_lambda=0.35'


# load_precomputed_vixmat= [r'77.0days']#[r'49.0days', r'105.days', r'140.days', r'259.0days']
# load_precomputed_dir= [load_dir_joint1] #[load_dir_joint1, load_dir_joint2, load_dir_joint3, load_dir_joint4]
# load_precomputed_file= ["ell_optimal_first_joint.npy"] #["ell_optimal_first_joint.npy","ell_optimal_first_joint.npy","ell_optimal_first_joint.npy","ell_optimal_first_joint.npy"]


# index_sel_maturities_spx=[8]
# index_sel_maturities_vix=[3]

#############################################

flag_hyper_both=False # If falls FALSE no Pre-search of the initial value #
flag_do_first_optim=True

take_last_ell=True #takes ell of previous slice!



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


# In[343]:
day=r'/20210602'


if config=="config8":
    #print("I am here", flush=True)
    maturities_vix=[r'14.0days',r'28.0days',r'49.0days',r'77.0days',r'105.0days',r'140.0days',r'259.0days']
    #maturities_spx=[r'16.0days',r'44.0days',r'58.0days',r'79.0days',r'107.0days',r'135.0days',r'170.0days',r'289.0days']
    maturities_spx=[r'14.0days',r'44.0days',r'58.0days',r'79.0days',r'107.0days',r'135.0days',r'170.0days', r'181.0days',r'198.0days',r'212.0days',r'233.0days',r'289.0days']
    
      
    maturities_joint=[r'14.0days',r'28.0days',r'44.0days',r'49.0days',r'58.0days',r'77.0days',r'79.0days',r'105.0days',r'107.0days',r'135.0days',r'140.0days',r'170.0days', r'181.0days',r'198.0days',r'212.0days',r'233.0days',r'259.0days',r'289.0days']
    
    
    moneyness_upperdev_vix=[1.2,1.2,2.3,3,3,3,3.5,3.8]
    moneyness_lowerdev_vix=[0.1,0.1,0.2,0.2,0.2,0.2,0.2]
        
    #guyon version
    moneyness_upperdev_spx=[0.05,0.1,0.2,0.25,0.3,0.35,0.35,0.35,0.35,0.35,0.35,0.5]
    moneyness_lowerdev_spx=[0.08,0.2,0.2,0.25,0.3,0.4,0.4,0.4,0.4,0.4,0.4,0.5]
    nbr_strikes_spx=[80,80,80,100,120,110,50,60,120,110,50,60]
else:
    maturities_vix=[r'14.0days',r'28.0days',r'49.0days',r'77.0days',r'105.0days',r'140.0days',r'259.0days']
    maturities_spx=[r'14.0days',r'44.0days',r'58.0days',r'79.0days',r'107.0days',r'135.0days',r'170.0days',r'289.0days']

    maturities_joint=[r'14.0days',r'28.0days',r'44.0days',r'49.0days',r'58.0days',r'77.0days',r'79.0days',r'105.0days',r'107.0days',r'135.0days',r'140.0days',r'170.0days',r'259.0days',r'289.0days']


    moneyness_upperdev_vix=[1.2,1,2.3,3,3,3,3,3.8] #change this to focus on a certain region of the smiles
    moneyness_lowerdev_vix=[0.1,0.1,0.1,0.2,0.2,0.2,0.2]
    
    moneyness_upperdev_spx=[0.035,0.05,0.1,0.15,0.3,0.35,0.35,0.35] #check this again
    moneyness_lowerdev_spx=[0.035,0.3,0.2,0.4,0.3,0.4,0.4,0.4]
    nbr_strikes_spx=[80,80,80,100,120,110,50,60]

# In[344]:





# In[345]:


os.chdir(r'/scratch.global/ag_cu/Data/VIX/Processed Data'+day) #directory load VIX data locally

list_strikes_vix=[]
list_prices_vix=[]
list_of_maturities_vix=[]
list_bid_vix=[]
list_ask_vix=[]
list_spot_vix=[]

for i,element in enumerate(maturities_vix):
    maturity_vix=element
    df_vix=pd.read_csv('ivol_data_maturity_'+maturity_vix)
    df_vix['strike']=df_vix['strike']/1000
    df_vix['open_interest']=df_vix['open_interest']/100 #not useful?
    scaling_flag_and_cut=True
    spot=df_vix['spot'][0]

    
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
        
        
    maturity=int(maturity_vix.split('.')[0])/365.25
    list_of_maturities_vix.append(maturity)
    list_strikes_vix.append(strikes_vix)
    list_prices_vix.append(prices_vix)
    list_spot_vix.append(spot)

list_of_maturities_vix=np.array(list_of_maturities_vix)
nbr_maturities_vix=len(list_of_maturities_vix)









os.chdir(r'/scratch.global/ag_cu/Data/SPX/Processed Data'+day) #directory load SPX data locally

list_bid_spx=[]
list_ask_spx=[]
list_strikes_spx=[]
list_prices_spx=[]
list_of_maturities_spx=[]


#nbr_strikes_spx=[80,100,80,110,120,110,50,60]

for i,element in enumerate(maturities_joint):
    
    if element in maturities_spx:
        maturity_spx=element
        idx=np.copy(i)
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
        
        if idx==index_sel_maturities_spx[0]:
            integers=np.array([int(b) for b in np.linspace(0,len(strikes_spx)-1,len(list_strikes_vix[index_sel_maturities_vix[0]]))])
        else:
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



flag_mat=True
N=int(maturity*365.25) # put here the daily sampling
T=maturity # put here maturities
#redundants



if int(config[-1])==1:
    d=3
    order_signature=2
    sigmas=[0.7,10,1]
    kappas=[0.1,25,0]
    X0=[1,0.08,0]
    thetas=[0.1,4,0]
    rng = np.random.default_rng(1267)
    Rho = random_correlation.rvs((2, 0.7, 0.3), random_state=rng)
if int(config[-1])==2:
    d=4
    order_signature=2
    sigmas=[0.7,10,5,1]
    kappas=[0.1,25,10,0]
    X0=[1,0.08,2,0]
    thetas=[0.1,4,0.08,0]
    os.chdir(r'/scratch.global/ag_cu/Codes_'+user+'/Randomness/n=2/config2')
    Rho =np.load('Rho_d=4.npy')
    
if int(config[-1])==3:  
    d=5
    order_signature=2
    sigmas=[0.7,10,5,6,1]
    kappas=[0.1,25,10,0.5,0]
    X0=[1,0.08,2,3,0]
    thetas=[0.1,4,0.08,0.5,0]
    rng = np.random.default_rng(13)
    Rho = random_correlation.rvs((3, 1, 0.7, 0.2,0.1), random_state=rng)
    
if int(config[-1])==4:
    d=3
    order_signature=3
    sigmas=[0.7,10,1]
    kappas=[0.1,25,0]
    X0=[1,0.08,0]
    thetas=[0.1,4,0]
    rng = np.random.default_rng(1267)
    Rho = random_correlation.rvs((2, 0.7, 0.3), random_state=rng)
      
if int(config[-1])==5:  
    d=6
    order_signature=2
    sigmas=[0.7,10,5,6,20,1]
    kappas=[0.1,25,10,0.5,10,0]
    X0=[1,0.08,2,3,0.6,0]
    thetas=[0.1,4,0.08,0.5,2,0]
    rng = np.random.default_rng(14)
    Rho = random_correlation.rvs((3, 1.5, 0.7,0.5, 0.2,0.1), random_state=rng)
    
if int(config[-1])==6:  
    d=7
    order_signature=2
    sigmas=[0.7,10,5,6,20,20,1]
    kappas=[0.1,25,10,0.5,10,20,0]
    X0=[1,0.08,2,3,0.6,5,0]
    thetas=[0.1,4,0.08,0.5,2,1,0]
    rng = np.random.default_rng(11)
    Rho = random_correlation.rvs((3, 1.5,1, 0.7,0.5, 0.2,0.1), random_state=rng)
    
if int(config[-1])==7:  
    d=8
    order_signature=2
    sigmas=[0.7,10,5,6,20,20,2,1]
    kappas=[0.1,25,10,0.5,10,20,10,0]
    X0=[1,0.08,2,3,0.6,5,3,0]
    thetas=[0.1,4,0.08,0.5,2,1,10,0]
    rng = np.random.default_rng(12)
    Rho = random_correlation.rvs((3, 1.5,1,0.9, 0.7,0.5, 0.3,0.1), random_state=rng)
if int(config[-1])==9 or int(config[-1])==8:  
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

os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness/n='+str(order_signature)+'/'+config)

if config[0]=="r":
    Q_0_all_truncated= np.load("rand_Q_0_all_truncated.npy")
    e_B_sig=np.load("rand_e_B_sig.npy")
    L_torch=np.load("rand_L_torch.npy")
    
    Q_0_all_truncated=torch.tensor(Q_0_all_truncated)
    e_B_sig= torch.tensor(e_B_sig)
    L_torch= torch.tensor(L_torch)
    
    param_dim=Q_0_all_truncated.shape[-1]

else:

    files=[]
    for element in os.listdir():
        if element!="JointCalibrationSamplingMaturities_CV_exact.py":
            files.append(element)
    
    
    param_dim=number_of_parameters_gen(order_signature,d)
    MC_nbr=80000
    
    Q_0_all_truncated=[]
    e_B_sig=[]
    L=[]
    
    for file in files:
        auxiliary=np.load(file,allow_pickle=True)
        if file[0]=='E':
            e_B_sig.append(auxiliary[:(index_sel_maturities_spx[0]+1),:,:])
        if file[0]=='L':
            L.append(auxiliary)
        if file[0]=='Q':
            Q_0_all_truncated.append(auxiliary[:(index_sel_maturities_spx[0]+1),:,:,:])
            
            
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
###############################################################




# def aux_torch(K,index_sel_maturities,norm,r=0,mat=None):
    
#     '''
#     Input: K (float); strike
#           index_sel_maturities, list (of indices corresponding to the desired maturities, e.g. [0] for the first mat)
#           norm: VIX_T or S_T; torch.tensors (outputs of VIX_T and get_S_T functions)
#     Outuput: matrix of payoffs (for one strike and all samples)
    
#     '''
#     if np.all(r==0):
#         matrix_=[torch.maximum(norm[j,:] - K,torch.tensor(0)) for j in range(len(index_sel_maturities))]
#         matrix=torch.stack(matrix_)
#         return matrix
#     else:
#         matrix_=[torch.maximum(np.exp(r[j]*mat[j])*norm[j,:] - K,torch.tensor(0)) for j in range(len(index_sel_maturities))]
#         matrix=torch.stack(matrix_)
#         return matrix

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
    
    #carefull!!! this seems to be worng because this sums over the number of samples
    #res=torch.sum(scalar_product,1)
    
    #we want to sum over the last dimension!
    res=torch.sum(scalar_product,-1)
    
    
    #J: below is the scaling without 100
    #norm=torch.sqrt(annualization*res)/(100) # HERE ILLEGAL SCALING APPLIED (?) 1/\delta
    
    #J: below is the scaling with 100
    norm=torch.sqrt(annualization*res)
    
    #J: below is no scaling at all,
    #norm=torch.sqrt(res)
    
    norm=norm.unsqueeze(0)
    return norm


# def monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities): #Cholesky decomposition here is used to speed this up
#     '''
#     Pricing function for the VIX, it runs for all strikes the payoff evaluation via aux_torch
#     '''

#     mc_payoff_arr=[aux_torch(K,index_sel_maturities,VIX) for K in strikes_vix]
#     return torch.stack(mc_payoff_arr).squeeze(1)

def monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities,r,mat,q=0): #Cholesky decomposition here is used to speed this up
    '''
    Pricing function for the VIX, it runs for all strikes the payoff evaluation via aux_torch
    '''
    #l=torch.from_numpy(l).float()
    #q is set to zero as no dividends are in the vix options
    #q=np.zeros(r.shape)
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



# def controlled_price(optimal_gammas,monte_carlo):
#     ''' 
#     Controlled price via MC-CV
#     '''
#     diff=monte_carlo-optimal_gammas
#     diff=torch.mean(diff,1)
#     return diff.numpy()

def controlled_price(monte_carlo): #optimal_gammas,
    ''' 
    Controlled price via MC-CV
    '''
    diff=monte_carlo #-optimal_gammas
    diff=torch.mean(diff,1)
    return diff.numpy()




Delta=1/12
annualization=(100**2)/Delta




# # Pricing of SPX

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



# def monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities,r,mat): #Cholesky decomposition here is NOT used 
#     '''
#     Pricing function for the SPX, it runs for all strikes the payoff evaluation via aux_torch
#     '''

#     mc_payoff_arr=[aux_torch(K,index_sel_maturities,S_T,r,mat) for K in strikes_spx]
#     return torch.stack(mc_payoff_arr).squeeze(1)

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
threshold_spx=100  # maybe do it vix/spx dependent + dependent on maturities and strikes?
threshold_vix=25
bid_ask_surface_spx=[]

list_rate_spx=[]
list_divi_spx=[]


ask_ivol_spx=[]
bid_ivol_spx=[]
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
        #las_vegas= las_vegas/np.mean(las_vegas)
        print("las_vegas SPX:", las_vegas)
        
        plt.plot(strikes_spx,np.array(iv_per_mat),color='blue')
        plt.scatter(strikes_spx,np.array(bid_per_mat),marker='+',color='red',label='Bid')
        plt.scatter(strikes_spx,np.array(ask_per_mat),marker='+',color='green',label='Ask')
        plt.legend()
        plt.title('T= '+element) ####################### ADD HERE SAVE FIG to save the smiles and bid-ask for SPX
        plt.show()
        
        plt.plot(strikes_spx,las_vegas,color='blue',marker='o',alpha=0.7)
        plt.title('Vega weights T= '+element)
        plt.show() ####################### ADD HERE SAVE FIG to save Las Vegas weights for SPX    
        
        las_vegas_spx.append(las_vegas)
        iv_surface_spx.append(np.array(iv_per_mat))
        bid_ask_surface_spx.append(np.array([bid_per_mat,ask_per_mat]))
        
        ask_ivol_spx.append(np.array(ask_per_mat))
        bid_ivol_spx.append(np.array(bid_per_mat))

        
    else:
        iv_surface_spx.append(-99)
        las_vegas_spx.append(-99)

        bid_ask_surface_spx.append(-99)

        list_rate_spx.append(-99)
        list_divi_spx.append(-99)
        
        ask_ivol_spx.append(-99)
        bid_ivol_spx.append(-99)

    
list_rate_spx=np.array(list_rate_spx)
list_divi_spx=np.array(list_divi_spx)

os.chdir(r'/scratch.global/ag_cu/Data/VIX/Processed Data'+day)

iv_surface_vix=[]
las_vegas_vix=[]
las_deltas_vix=[] 
bid_ask_surface_vix=[]
list_rate_vix=[]

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
    list_rate_vix.append(rate_vix)
    
    ask_ivol_vix.append(np.array(ask_per_mat))
    bid_ivol_vix.append(np.array(bid_per_mat))
        





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




if len(load_precomputed_dir)!=0:
    l_history=[]
    for i, d in enumerate(load_precomputed_dir):
        os.chdir(d)
        loaded_ell= np.load(load_precomputed_file[i])
        l_history.append(torch.tensor(loaded_ell))
        
    
    idx_mat_loaded=[]
    for m, mat in enumerate(maturities_joint):
        if mat in load_precomputed_vixmat:
            idx_mat_loaded.append(m)
    
    correction=0
    for i in range(len(idx_mat_loaded)):
        if i==0:
            Q0_corr=Q_0_all_truncated[idx_mat_loaded[i]]
            linear_sig_corr=e_B_sig[idx_mat_loaded[i]]
            correction+= -0.5*quadratic_form(l_history[i],Q0_corr)+torch.matmul(linear_sig_corr,l_history[i])
        else:
            Q0_corr=Q_0_all_truncated[idx_mat_loaded[i]]- Q_0_all_truncated[idx_mat_loaded[i-1]]
            linear_sig_corr=e_B_sig[idx_mat_loaded[i]]- e_B_sig[idx_mat_loaded[i-1]]
            correction+= -0.5*quadratic_form(l_history[i],Q0_corr)+torch.matmul(linear_sig_corr,l_history[i])
else:
    correction=0

nbr_options_spx=len(list_prices_spx[index_sel_maturities_spx[0]])
prices_spx=list_prices_spx[index_sel_maturities_spx[0]]


#weights_spx=np.sqrt(las_vegas_spx[index_sel_maturities_spx[0]])
weights_spx=np.array([1 for _ in range(len(las_vegas_spx[index_sel_maturities_spx[0]]))])
strikes_spx=list_strikes_spx[index_sel_maturities_spx[0]]


prices_vix=list_prices_vix[index_sel_maturities_vix[0]]
#weights_vix=np.sqrt(las_vegas_vix[index_sel_maturities_vix[0]])
weights_vix=np.array([1 for _ in range(len(las_vegas_vix[index_sel_maturities_vix[0]]))])

strikes_vix=list_strikes_vix[index_sel_maturities_vix[0]]

    

def get_ST_timevarying(l,correction,m,Q_0_all_truncated,e_B_sig,c):
    idx_to_calib=m
    
    if m==0:
        Q0=Q_0_all_truncated[idx_to_calib]
        linear_sig=e_B_sig[idx_to_calib]
        log_ST=-0.5*quadratic_form(l,Q0)+torch.matmul(linear_sig,l)
        
    else: # so far only implemented for exactly this maturity!
        
        Q0_current=(Q_0_all_truncated[m]-Q_0_all_truncated[idx_mat_loaded[-1]]) #(add and do not add to calibration)
        linear_sig_current=(e_B_sig[m]-e_B_sig[idx_mat_loaded[-1]])*c
        
        
        log_ST=-0.5*quadratic_form(l,Q0_current)+torch.matmul(linear_sig_current,l)
        log_ST=log_ST+correction

    return torch.exp(log_ST)


    
print("before initial samples", flush=True)

if flag_hyper_both==True:
    #c=100
    
    nbr_samples_ell=10000
    
    Q_0_all_truncated_only_rel= Q_0_all_truncated[:(index_sel_maturities_spx[0]+1), :, :, :]
    e_B_sig_only_rel= e_B_sig[:(index_sel_maturities_spx[0]+1), :, :]
    L_torch_only_rel= L_torch[index_sel_maturities_vix[0], :, :, :]
    
        
    
    ell_sampled=[]
    error_spx = []
    error_vix = []
    error_joint=[]
    sampled_prices_spx= []
    sampled_prices_vix= []
    
    bid_vix= list_bid_vix[index_sel_maturities_vix[0]]
    ask_vix= list_ask_vix[index_sel_maturities_vix[0]]
    bid_ask_spread_vix= bid_vix-ask_vix
    
    bid_spx= list_bid_spx[index_sel_maturities_spx[0]]
    ask_spx= list_ask_spx[index_sel_maturities_spx[0]]
    bid_ask_spread_spx= bid_spx-ask_spx

    
    #for j in tqdm(range(nbr_samples_ell),desc='Sampling initial values SPX'):
    for j in range(nbr_samples_ell):
        #print("here", j, flush=True)
        #rng = np.random.default_rng(6394829)
        #ell=torch.tensor(uniform.rvs(loc=-0.002, scale=0.004, size=Q_0_all_truncated.shape[-1], random_state=rng))
        #ell=torch.tensor(np.random.uniform(-0.002,0.002,Q_0_all_truncated_only_rel.shape[-1])) #to be tuned
        #ell=torch.tensor(np.random.uniform(-0.01,0.01,Q_0_all_truncated_only_rel.shape[-1]))
        ell=torch.tensor(np.random.uniform(-0.1,0.1,Q_0_all_truncated_only_rel.shape[-1]))

        ell_sampled.append(ell)
        #print("sampled ell", flush=True)
        #SPX
        S_T=get_ST_timevarying(ell,correction,index_sel_maturities_spx[0],Q_0_all_truncated_only_rel,e_B_sig_only_rel,1)    
        S_T=S_T.unsqueeze(0)
        
        #optimal_gammas=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)

        idx= index_sel_maturities_spx[0]
        monte_carlo=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
        spx_model_prices=controlled_price(monte_carlo)
        #print("calculated SPX", flush=True)

        VIX=VIX_T(ell,L_torch,annualization)
        VIX=VIX[:,index_sel_maturities_vix[0],:]
        idx=index_sel_maturities_vix[0]
        #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
        monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
        vix_model_prices=controlled_price(monte_carlo_vix)
        
        VIX_future= torch.mean(VIX) #*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))
        #print("calculated VIX", flush=True)
        
        #print(VIX_future, flush=True)
        
        sampled_prices_spx.append(spx_model_prices)
        sampled_prices_vix.append(vix_model_prices)
        #print("stored prices", flush=True)
        err_spx=np.sum(np.abs((prices_spx-spx_model_prices)/bid_ask_spread_spx)) ##add scaling of bid_ask
        error_spx.append(err_spx) 
        #print("stored error spx", flush=True)
        err_vix=np.sum(np.abs((prices_vix-vix_model_prices)/bid_ask_spread_vix))
        error_vix.append(err_vix) 
        #print("stored error vix", flush=True)
        
        # print("0", (list_spot_vix[index_sel_maturities_vix[0]]), flush=True)
        # print("1",list_spot_vix[index_sel_maturities_vix[0]]-VIX_future.numpy() , flush=True)
        # print("2", np.abs(list_spot_vix[index_sel_maturities_vix[0]]-VIX_future.numpy()), flush=True)
        # print("3", (np.abs(list_spot_vix[index_sel_maturities_vix[0]]-VIX_future.numpy()))/(list_spot_vix[index_sel_maturities_vix[0]]), flush=True)
        
        
        err_future= (np.abs(list_spot_vix[index_sel_maturities_vix[0]]-VIX_future.numpy()))/(list_spot_vix[index_sel_maturities_vix[0]])
        #print("calculated error future", flush=True)
        
        error_joint.append(err_vix+err_spx+ err_future)
        #print("stored joint error", flush=True)
        
    sampled_prices_spx=np.array(sampled_prices_spx)
    sampled_prices_vix=np.array(sampled_prices_vix)
    
    error_spx=np.array(error_spx)
    error_vix=np.array(error_vix)
    error_joint=np.array(error_joint)
    
        
   
    idx_min =np.where( error_joint==np.min( error_joint))[0][0]
    
    
    
    print('Minimal error on BOTH MC:', error_joint[idx_min])
    print('Minimal error on BOTH for SPX:', error_spx[idx_min])
    print('Minimal error on BOTH for VIX:', error_vix[idx_min])
    
    
    #c_init= c_sampled[row_idx_min]
    ell_init=ell_sampled[idx_min]

    S_T=get_ST_timevarying(ell_init,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1)    
    S_T=S_T.unsqueeze(0)
    #optimal_gammas=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx= index_sel_maturities_spx[0]
    monte_carlo=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    init_spx_model_prices=controlled_price(monte_carlo)
    
    
    VIX=VIX_T(ell_init,L_torch,annualization)
    VIX=VIX[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    init_vix_model_prices=controlled_price(monte_carlo_vix)
    VIX_future= torch.mean(VIX)#*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))

 
    check_VIX=VIX_T(ell_init,L_torch_only_rel,annualization)
    #VIX=VIX[:,index_sel_maturities_vix[0],:]
    #check_optimal_gammas_vix=torch.stack([optimal_controls(check_VIX,strike) for strike in strikes_vix]).squeeze(1)
    check_monte_carlo_vix=monte_carlo_pricing_vix(check_VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    check_init_vix_model_prices=controlled_price(check_monte_carlo_vix)
 
    if len(load_precomputed_dir)>0:
        save_init=load_precomputed_dir[-1]+ "/ROLLING/Maturity_SPX"+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config+"/Initial"
    else:
        save_init= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/Maturity_SPX'+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config
    os.makedirs(save_init, exist_ok=True)
    os.chdir(save_init)
    
    plt.figure()
    plt.plot(strikes_spx,init_spx_model_prices,marker='o',label='Model')
    plt.plot(strikes_spx,prices_spx,marker='*',label='Market')
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('SPX')
    plt.savefig("SPX")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_spx,sampled_prices_spx[idx_min],marker='o',label='Model')
    plt.plot(strikes_spx,prices_spx,marker='*',label='Market')
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('SPX')
    plt.savefig("SPX_alt")
    plt.show()
    
    
    plt.figure()
    plt.plot(strikes_spx,np.abs(init_spx_model_prices-prices_spx),marker='o')
    plt.xlabel('Strikes')
    plt.title('Absolute Error SPX')
    plt.savefig("Error SPX")
    plt.show()
    
    
    
    plt.figure()
    plt.plot(strikes_vix,init_vix_model_prices,marker='o',label='Model')
    plt.plot(strikes_vix,prices_vix,marker='*',label='Market')
    plt.axvline(x=list_spot_vix[index_sel_maturities_vix[0]], ls="--", label="market future", color="red")
    plt.axvline(x=VIX_future, ls="--", label="model future", color="blue")
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('VIX')
    plt.savefig("VIX")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,check_init_vix_model_prices,marker='o',label='Model')
    plt.plot(strikes_vix,prices_vix,marker='*',label='Market')
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('check_VIX')
    plt.savefig("check_VIX")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,sampled_prices_vix[idx_min],marker='o',label='Model')
    plt.plot(strikes_vix,prices_vix,marker='*',label='Market')
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('VIX')
    plt.savefig("VIX_alt")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,np.abs(init_vix_model_prices-prices_vix),marker='o')
    plt.xlabel('Strikes')
    plt.title('Absolute Error VIX')
    plt.savefig("Error VIX")
    plt.show()
    # ell_init_vix=ell_sampled[idx_min_vix] # best initial ell for the vix
    # ell_init_spx=ell_sampled2[idx_min_spx] # best initial ell for the spx
    
    with open('INFO.txt', 'w') as f:

        f.write(r"load_precomputed_vixmat: "+str(load_precomputed_vixmat)+ os.linesep)
        f.write(r"load_precomputed_dir: "+str(load_precomputed_dir)+ os.linesep)
        f.write(r"load_precomputed_file: "+str(load_precomputed_file)+ os.linesep)

    
    np.save('ell_init_both.npy',ell_init)
    #np.save('c_init.npy',c_init)
    #np.save('ell_init_SPX2.npy',ell_init_spx)


elif not take_last_ell:
    if len(load_precomputed_dir)>0:
        save_init=load_precomputed_dir[-1]+ "/ROLLING/Maturity_SPX"+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config+"/Initial"
    else:
        save_init= r'/scratch.global/ag_cu/Codes_'+user+'/optimal_initial_values/Maturity_SPX'+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config

    os.chdir(save_init)
    
    ell_init= np.load('ell_init_both.npy')
    #c_init= np.load('c_init.npy')
    
    ell_init= torch.tensor(ell_init)
    #c_init= c_init[0]
    
if take_last_ell:
    ell_init= l_history[-1]


def soft_indicator(x):
    return 0.5*(np.tanh(x*100)+1)

def loss_vix(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)#[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    diff2=np.multiply(prices_vix-controlled_vix,weights_vix)
    return diff2


def loss_vix_BA(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    diff2=np.multiply(prices_vix-controlled_vix,weights_vix)
    diff2=np.divide(diff2,prices_vix-bid_ask_surface_vix[0][0]) #### ADDED this
    return diff2

def loss_vix_BA_v2(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    diff2=prices_vix-controlled_vix
    diff2=np.divide(diff2,prices_vix-bid_ask_surface_vix[0][0]) #### ADDED this
    return diff2

bid_vix= list_bid_vix[index_sel_maturities_vix[0]]
ask_vix= list_ask_vix[index_sel_maturities_vix[0]]

def funny_loss_vix(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])

    controlled_vix=controlled_price(monte_carlo_vix)
    indicator= [(controlled_vix[k]<=bid_vix[k] or controlled_vix[k]>=ask_vix[k]) for k in range(len(strikes_vix))]
    diff2=np.multiply(prices_vix-controlled_vix,weights_vix)*indicator
    return diff2

def loss_vix_soft_indicator(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    
    VIX_future= torch.mean(VIX)#*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))

    
    #indicator= [(controlled_vix[k]<=bid_vix[k] or controlled_vix[k]>=ask_vix[k]) for k in range(len(strikes_vix))]
    #diff2=np.multiply((prices_vix-controlled_vix)**2,weights_vix**2)*(soft_indicator(bid_vix-controlled_vix)+soft_indicator(controlled_vix-ask_vix))
    diff2=np.multiply((prices_vix-controlled_vix)**power,(1/(np.abs(bid_vix-ask_vix))*weights_vix)**power)*(soft_indicator(bid_vix-controlled_vix)+soft_indicator(controlled_vix-ask_vix))
    diff2=np.mean(diff2)
    diff2+=10*(VIX_future.numpy()-list_spot_vix[index_sel_maturities_vix[0]])**power
    return diff2

def loss_vix_soft_indicator_vega(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    
    VIX_future= torch.mean(VIX)#*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))

    
    #indicator= [(controlled_vix[k]<=bid_vix[k] or controlled_vix[k]>=ask_vix[k]) for k in range(len(strikes_vix))]
    #diff2=np.multiply((prices_vix-controlled_vix)**2,weights_vix**2)*(soft_indicator(bid_vix-controlled_vix)+soft_indicator(controlled_vix-ask_vix))
    diff2=np.multiply((prices_vix-controlled_vix)**power,(1/(np.abs(bid_ivol_vix[index_sel_maturities_vix[0]]-ask_ivol_vix[index_sel_maturities_vix[0]]))*las_vegas_vix[index_sel_maturities_vix[0]])**power)*(soft_indicator(bid_vix-controlled_vix)+soft_indicator(controlled_vix-ask_vix))
    diff2=np.mean(diff2)
    diff2+=10*(VIX_future.numpy()-list_spot_vix[index_sel_maturities_vix[0]])**power
    return diff2


def loss_vix_linear_vega_delta(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    
    VIX_future= torch.mean(VIX)#*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))

        
    vix_future_arr= np.array([VIX_future.numpy()]*len(list_strikes_vix[index_sel_maturities_vix[0]]))
    vix_spot_arr= np.array([list_spot_vix[index_sel_maturities_vix[0]]]*len(list_strikes_vix[index_sel_maturities_vix[0]]))
    #indicator= [(controlled_vix[k]<=bid_vix[k] or controlled_vix[k]>=ask_vix[k]) for k in range(len(strikes_vix))]
    #diff2=np.multiply((prices_vix-controlled_vix)**2,weights_vix**2)*(soft_indicator(bid_vix-controlled_vix)+soft_indicator(controlled_vix-ask_vix))
        
        ##diff_1= np.multiply((list_prices_vix[idx]-controlled_vix)**2,((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx])**2)
    
    diff_1= np.abs(np.multiply((list_prices_vix[index_sel_maturities_vix[0]]-controlled_vix),((1/np.abs(bid_ivol_vix[index_sel_maturities_vix[0]]-ask_ivol_vix[index_sel_maturities_vix[0]]))*las_vegas_vix[index_sel_maturities_vix[0]])))
    diff_2= np.abs(np.multiply((vix_spot_arr-vix_future_arr),((1/np.abs(bid_ivol_vix[index_sel_maturities_vix[0]]-ask_ivol_vix[index_sel_maturities_vix[0]]))*las_vegas_vix[index_sel_maturities_vix[0]]*las_deltas_vix[index_sel_maturities_vix[0]])))

    diff= np.mean((diff_1+diff_2)**power)

    return diff

def loss_vix_softindicator_vega_delta(l):
    l=torch.tensor(l)
    idx=index_sel_maturities_vix[0]
    VIX=VIX_T(l,L_torch[idx,:,:,:],annualization)
    #VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    #idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
       
    VIX_future= torch.mean(VIX)#*torch.tensor(np.exp(-1*list_rate_vix[idx]*list_of_maturities_vix[idx]))
        
    vix_future_arr= np.array([VIX_future.numpy()]*len(list_strikes_vix[idx]))
    vix_spot_arr= np.array([list_spot_vix[idx]]*len(list_strikes_vix[idx]))

    diff_1= np.abs(np.multiply((list_prices_vix[idx]-controlled_vix),((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx])))*(soft_indicator(list_bid_vix[idx]-controlled_vix)+soft_indicator(controlled_vix-list_ask_vix[idx]))
    diff_2= np.abs(np.multiply((vix_spot_arr-vix_future_arr),((1/np.abs(bid_ivol_vix[idx]-ask_ivol_vix[idx]))*las_vegas_vix[idx]*las_deltas_vix[idx])))

    diff= np.mean((diff_1+diff_2)**power)

    return diff





def loss_spx(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    #indicator= (controlled_spx<=bid_spx and controlled_vix>=ask_spx)
    diff1=np.multiply(prices_spx-controlled_spx,weights_spx)
    return diff1


def loss_spx_BA(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])

    controlled_spx=controlled_price(monte_carlo_spx)
    #indicator= (controlled_spx<=bid_spx and controlled_vix>=ask_spx)
    diff1=np.multiply(prices_spx-controlled_spx,weights_spx)
    diff1=np.divide(diff1,prices_spx-bid_ask_surface_spx[2][0]) #### ADDED this
    return diff1


def loss_spx_BA_v2(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    #indicator= (controlled_spx<=bid_spx and controlled_vix>=ask_spx)
    diff1=prices_spx-controlled_spx
    diff1=np.divide(diff1,prices_spx-bid_ask_surface_spx[2][0]) #### ADDED this
    return diff1


bid_spx= list_bid_spx[index_sel_maturities_spx[0]]
ask_spx= list_ask_spx[index_sel_maturities_spx[0]]

#aux_weights_spx= [1,1,1,1,1,1,100,100,100,1,1,1]

def funny_loss_spx(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    indicator= [(controlled_spx[k]<=bid_spx[k] or controlled_spx[k]>=ask_spx[k]) for k in range(len(strikes_spx))]
    diff1=np.multiply(prices_spx-controlled_spx,weights_spx)*indicator
    return diff1

def loss_spx_soft_indicator(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    #indicator= [(controlled_spx[k]<=bid_spx[k] or controlled_spx[k]>=ask_spx[k]) for k in range(len(strikes_spx))]
    #diff1=np.multiply((prices_spx-controlled_spx)**2,weights_spx**2)*(soft_indicator(bid_spx-controlled_spx)+soft_indicator(controlled_spx-ask_spx))
    diff1=np.multiply((prices_spx-controlled_spx)**power,(1/(np.abs(bid_spx-ask_spx))*weights_spx)**power)*(soft_indicator(bid_spx-controlled_spx)+soft_indicator(controlled_spx-ask_spx))
    return np.mean(diff1)

def loss_spx_soft_indicator_vega(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    #indicator= [(controlled_spx[k]<=bid_spx[k] or controlled_spx[k]>=ask_spx[k]) for k in range(len(strikes_spx))]
    #diff1=np.multiply((prices_spx-controlled_spx)**2,weights_spx**2)*(soft_indicator(bid_spx-controlled_spx)+soft_indicator(controlled_spx-ask_spx))
    diff1=np.multiply((prices_spx-controlled_spx)**power,(1/(np.abs(bid_ivol_spx[index_sel_maturities_spx[0]]-ask_ivol_spx[index_sel_maturities_spx[0]]))*las_vegas_spx[index_sel_maturities_spx[0]])**power)*(soft_indicator(bid_spx-controlled_spx)+soft_indicator(controlled_spx-ask_spx))
    return np.mean(diff1)

def loss_spx_linear_vega(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    #indicator= [(controlled_spx[k]<=bid_spx[k] or controlled_spx[k]>=ask_spx[k]) for k in range(len(strikes_spx))]
    #diff1=np.multiply((prices_spx-controlled_spx)**2,weights_spx**2)*(soft_indicator(bid_spx-controlled_spx)+soft_indicator(controlled_spx-ask_spx))
    diff1=np.multiply((prices_spx-controlled_spx)**power,(1/(np.abs(bid_ivol_spx[index_sel_maturities_spx[0]]-ask_ivol_spx[index_sel_maturities_spx[0]]))*las_vegas_spx[index_sel_maturities_spx[0]])**power)
    return np.mean(diff1)

def loss_spx_softindicator_vega(l): #now with time-varying function for S_T, m=1 as it is the first joint calibration
    l=torch.tensor(l)
    #diff=0
    #for i, idx in enumerate(index_sel_maturities_spx):
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
        #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[idx],list_of_maturities_spx[idx]) for strike in list_strikes_spx[idx]]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)

    #indicator= [(controlled_spx[k]<=bid_spx[k] or controlled_spx[k]>=ask_spx[k]) for k in range(len(strikes_spx))]
    #diff1=np.multiply((prices_spx-controlled_spx)**2,weights_spx**2)*(soft_indicator(bid_spx-controlled_spx)+soft_indicator(controlled_spx-ask_spx))
    diff=np.multiply((prices_spx-controlled_spx)**power,((1/np.abs(bid_ivol_spx[idx]-ask_ivol_spx[idx]))*las_vegas_spx[idx])**power)*(soft_indicator(list_bid_spx[idx]-controlled_spx)+soft_indicator(controlled_spx-list_ask_spx[idx]))
    return np.mean(diff)

#comp_price_scale= np.mean(prices_vix)/np.mean(prices_spx) # here I compensate for the different scale in prices for VIX and SPX
comp_price_scale=1


def loss_joint(l):
    diff1=loss_spx(l)
    diff2=loss_vix(l)
    res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
    return res



def loss_joint_BA(l):
    diff1=loss_spx_BA(l)
    diff2=loss_vix_BA(l)
    res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
    return res

def loss_joint_BA_v2(l):
    diff1=loss_spx_BA_v2(l)
    diff2=loss_vix_BA_v2(l)
    res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
    return res

def funny_loss_joint(l):
    diff1=funny_loss_spx(l)
    diff2=funny_loss_vix(l)
    res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
    return res

def loss_joint_soft_indicator(l):
    diff1=loss_spx_soft_indicator(l)
    diff2=loss_vix_soft_indicator(l)
    #res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
    res= lambda_coeff*diff1+ (1-lambda_coeff)*diff2
    return res   

def loss_joint_soft_indicator_vega(l):
    diff1=loss_spx_soft_indicator_vega(l)
    diff2=loss_vix_soft_indicator_vega(l)
    #res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
    res= lambda_coeff*diff1+ (1-lambda_coeff)*diff2
    return res    

def loss_joint_linear_vega_delta(l):
    
    if lambda_coeff==1:
        #print("here", flush=True)
        diff1=loss_spx_linear_vega(l)
        #diff2=loss_vix_linear_vega_delta(l)
        #res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
        #res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
        return diff1 
    elif lambda_coeff==0:
        #diff1=loss_spx_linear_vega(l)
        diff2=loss_vix_linear_vega_delta(l)
        #res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
        #res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
        return diff2
        
    else:
        diff1=loss_spx_linear_vega(l)
        diff2=loss_vix_linear_vega_delta(l)
        #res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
        res= lambda_coeff*(diff1)+ (1-lambda_coeff)*diff2
        return res 


def loss_joint_softindicator_vega_delta(l):
    diff1=loss_spx_softindicator_vega(l)
    diff2=loss_vix_softindicator_vega_delta(l)
    #res=np.concatenate([np.sqrt(comp_price_scale)*np.sqrt(lambda_coeff)*np.sqrt(1/len(strikes_spx))*diff1,np.sqrt(1-lambda_coeff)*np.sqrt(1/len(strikes_vix))*diff2],axis=0)
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

        if loss_flag=="LP":
            res= least_squares(loss_joint_BA, ell_init.numpy(),loss=rho)
        elif loss_flag=="LP_V2":
            res= least_squares(loss_joint_BA_v2, ell_init.numpy(),loss=rho_v2)
        elif loss_flag=="SOFT_INDICATOR":
            res= minimize(loss_joint_soft_indicator, ell_init.numpy())
        elif loss_flag=="SOFT_INDICATOR_VEGA":
            res= minimize(loss_joint_soft_indicator_vega, ell_init.numpy())
        elif loss_flag=="LINEAR":
            res= least_squares(loss_joint_BA, ell_init.numpy(),loss="linear")
        elif loss_flag=="LINEAR_VEGA_DELTA":
            res= minimize(loss_joint_linear_vega_delta, ell_init.numpy())#, options={'maxiter':10})
        elif loss_flag=="SOFT_INDICATOR_VEGA_DELTA":
            res= minimize(loss_joint_softindicator_vega_delta, ell_init.numpy())#, options={'maxiter':10})

      
    if len(load_precomputed_dir)>0:
        if take_last_ell:
            save_dir=load_precomputed_dir[-1]+ "/ROLLING/Maturity_SPX"+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config+r'/'+loss_flag+str(power)+'_lambda='+str(lambda_coeff)+r"/took_last"
        else:
            save_dir=load_precomputed_dir[-1]+ "/ROLLING/Maturity_SPX"+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config+r'/'+loss_flag+str(power)+'_lambda='+str(lambda_coeff)

    else:
        save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/Maturity_SPX'+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config+r'/'+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
    os.makedirs(save_dir, exist_ok=True)
    
    os.chdir(save_dir)
    

    np.save('ell_optimal_first_joint.npy',res['x'])
    
    l=torch.tensor(res['x'])
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    
    
    np.save('prices_optimal_SPX.npy',controlled_spx)
        
    
    VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    VIX_future= torch.mean(VIX)
    print("VIX_future without rate:", VIX_future)
    print("VIX_future with rate:", VIX_future*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]])))
    print("VIX spot:", list_spot_vix[index_sel_maturities_vix[0]])
    
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)

    np.save('prices_optimal_VIX.npy',controlled_vix)
    
    #VIX_future=VIX_future*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))
    np.save('optimal_VIX_futures.npy',VIX_future)
    
    plt.figure()
    plt.plot(strikes_spx,controlled_spx,marker='o',label='Model')
    plt.plot(strikes_spx,prices_spx,marker='*',label='Market')
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('SPX')
    #np.save('prices_optimal_VIX_lp.npy',controlled_vix)
    #if loss_flag=="LP" or loss_flag=="LP_V2":
    plt.savefig("Calibrated_SPX")

    plt.show()
    
    plt.figure()
    plt.plot(strikes_spx,np.abs(controlled_spx-prices_spx),marker='o')
    plt.xlabel('Strikes')
    plt.title('Absolute Error SPX')
    #if loss_flag=="LP" or loss_flag=="LP_V2":
    plt.savefig("Calibrated Error_SPX")

    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,controlled_vix,marker='o',label='Model')
    plt.plot(strikes_vix,prices_vix,marker='*',label='Market')
    plt.axvline(x=list_spot_vix[index_sel_maturities_vix[0]], ls="--", label="market future", color="red")
    plt.axvline(x=VIX_future, ls="--", label="model future", color="blue")
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('VIX')
    #if loss_flag=="LP" or loss_flag=="LP_V2":
    plt.savefig("Calibrated_VIX")
    
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,np.abs(controlled_vix-prices_vix),marker='o')
    plt.xlabel('Strikes')
    plt.title('Absolute Error VIX')
    #if loss_flag=="LP" or loss_flag=="LP_V2":
    plt.savefig("Calibrated Error_VIX")
    
    plt.show()
    
    
else: 
            
    save_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/Maturity_SPX'+str(index_sel_maturities_spx)+'_VIX'+str(index_sel_maturities_vix)+'/'+config+r'/'+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
    
    os.chdir(save_dir) 
        
    l= np.load('ell_optimal_first_joint.npy')
    l=torch.tensor(l)
    S_T=get_ST_timevarying(l,correction,index_sel_maturities_spx[0],Q_0_all_truncated,e_B_sig,1).unsqueeze(0)
    #optimal_gammas_spx=torch.stack([optimal_controls(S_T,strike,(list_rate_spx-list_divi_spx)[index_sel_maturities_spx[0]],list_of_maturities_spx[index_sel_maturities_spx[0]]) for strike in strikes_spx]).squeeze(1)
    idx=index_sel_maturities_spx[0]
    monte_carlo_spx=monte_carlo_pricing_spx_nochol(S_T,strikes_spx,index_sel_maturities_spx,list_rate_spx[idx],list_divi_spx[idx],list_of_maturities_spx[idx])
    controlled_spx=controlled_price(monte_carlo_spx)
    
    if loss_flag=="LP" or loss_flag=="LP_V2":
        np.save('prices_optimal_SPX_lp.npy',controlled_spx)
    elif loss_flag=="SOFT_INDICATOR" or loss_flag=="SOFT_INDICATOR_VEGA":
        np.save('prices_optimal_SPX_si.npy',controlled_spx)
    elif loss_flag=="LINEAR":
        np.save('prices_optimal_SPX_lin.npy',controlled_spx)
        
    
    VIX=VIX_T(l,L_torch,annualization)[:,index_sel_maturities_vix[0],:]
    VIX_future= torch.mean(VIX)
    print("VIX_future without rate:", VIX_future)
    print("VIX_future with rate:", VIX_future*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]])))
    print("VIX spot:", list_spot_vix[index_sel_maturities_vix[0]])
    
    #optimal_gammas_vix=torch.stack([optimal_controls(VIX,strike) for strike in strikes_vix]).squeeze(1)
    idx=index_sel_maturities_vix[0]
    monte_carlo_vix=monte_carlo_pricing_vix(VIX,strikes_vix,index_sel_maturities_vix,list_rate_vix[idx],list_of_maturities_vix[idx])
    controlled_vix=controlled_price(monte_carlo_vix)
    
    if loss_flag=="LP" or loss_flag=="LP_V2":
        np.save('prices_optimal_VIX_lp.npy',controlled_vix)
    elif loss_flag=="SOFT_INDICATOR" or loss_flag=="SOFT_INDICATOR_VEGA":
        np.save('prices_optimal_VIX_si.npy',controlled_vix)
    elif loss_flag=="LINEAR":
        np.save('prices_optimal_VIX_lin.npy',controlled_vix)
    
    #VIX_future=VIX_future*torch.tensor(np.exp(-1*list_rate_vix[index_sel_maturities_vix[0]]*list_of_maturities_vix[index_sel_maturities_vix[0]]))
    np.save('optimal_VIX_futures.npy',VIX_future)
    
    plt.figure()
    plt.plot(strikes_spx,controlled_spx,marker='o',label='Model')
    plt.plot(strikes_spx,prices_spx,marker='*',label='Market')
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('SPX')
    np.save('prices_optimal_VIX_lp.npy',controlled_vix)
    if loss_flag=="LP" or loss_flag=="LP_V2":
        plt.savefig("Calibrated_SPX_lp")
    elif loss_flag=="SOFT_INDICATOR" or loss_flag=="SOFT_INDICATOR_VEGA":
        plt.savefig("Calibrated_SPX_si")
    elif loss_flag=="LINEAR":
        plt.savefig("Calibrated_SPX_lin")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_spx,np.abs(controlled_spx-prices_spx),marker='o')
    plt.xlabel('Strikes')
    plt.title('Absolute Error SPX')
    if loss_flag=="LP" or loss_flag=="LP_V2":
        plt.savefig("Calibrated Error_SPX_lp")
    elif loss_flag=="SOFT_INDICATOR" or loss_flag=="SOFT_INDICATOR_VEGA":
        plt.savefig("Calibrated Error_SPX_si")
    elif loss_flag=="LINEAR":
        plt.savefig("Calibrated Error_SPX_lin")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,controlled_vix,marker='o',label='Model')
    plt.plot(strikes_vix,prices_vix,marker='*',label='Market')
    plt.axvline(x=list_spot_vix[index_sel_maturities_vix[0]], ls="--", label="market future", color="red")
    plt.axvline(x=VIX_future, ls="--", label="model future", color="blue")
    plt.xlabel('Strikes')
    plt.legend()
    plt.title('VIX')
    if loss_flag=="LP" or loss_flag=="LP_V2":
        plt.savefig("Calibrated_VIX_lp")
    elif loss_flag=="SOFT_INDICATOR" or loss_flag=="SOFT_INDICATOR_VEGA":
        plt.savefig("Calibrated_VIX_si")
    elif loss_flag=="LINEAR":
        plt.savefig("Calibrated_VIX_lin")
    plt.show()
    
    plt.figure()
    plt.plot(strikes_vix,np.abs(controlled_vix-prices_vix),marker='o')
    plt.xlabel('Strikes')
    plt.title('Absolute Error VIX')
    if loss_flag=="LP" or loss_flag=="LP_V2":
        plt.savefig("Calibrated Error_VIX_lp")
    elif loss_flag=="SOFT_INDICATOR" or loss_flag=="SOFT_INDICATOR_VEGA":
        plt.savefig("Calibrated Error_VIX_si")
    elif loss_flag=="LINEAR":
        plt.savefig("Calibrated Error_VIX_lin")
    plt.show()
        
    
