"""
Created on Sat Jan 28 18:08:38 2023

@author: Guido Gazzani & Janka Moeller
"""


import numpy as np
import pandas as pd
import itertools as itt  
from tqdm.auto import tqdm
import os
import torch
from numpy.linalg import cholesky
from joblib import Parallel,delayed
from scipy.stats import random_correlation
import signatory 

print('Signatory version:',signatory.__version__)
print('Torch version',torch.__version__)
print('Numpy version',np.__version__)

#Choose maturities and day (notice not the same maturities across different days!)

flag_gatheral=False
flag_missing_last=False
flag_save_control_variates=False
available_Gplus=False
numerical_approx=False
flag_chol_vix=True


########## the configuration is just an indicator of some pre-initialized parameters of the primary object
########## it acts as our Reservoir

config='config8'


if flag_gatheral==True:    
##################### for GATHERAL TEST (see the maturities used in the Quadratic Rough Heston)
    list_of_maturities_vix=np.array([12.0, 19.0, 26.0, 33.0])/365.25
    list_of_maturities_spx=np.array([12.0, 19.0, 26.0, 33.0])/365.25

else:
    list_of_maturities_vix=np.array([14.0, 28.0, 49.0, 77.0, 105.0, 140.0, 259.0])/365.25
    list_of_maturities_spx=np.array([14.0, 44.0, 58.0, 79.0, 107.0, 135.0,  170.0, 181.0, 198.0, 212.0, 233.0, 289.0, 380.0])/365.25

    

#For 20210602
############################################
# PLEASE SELECT MORE SPX MATURITIES THAN VIX MATURITIES!



def cubes_zeros(n,m):
    '''
    Inputs:
    n: int
    m: int
    
    Return: list of m+1 cubes of increasing order from 0 to n^m
    '''
    cubes=[0,]
    for K in range(0,m):
        cubes.append(np.tensordot(cubes[-1],np.zeros((n,)), axes=0))
    return cubes

def fromcubestoarray (cubes):
    '''
    Inputs:
    cubes: list of cubes of increasing order
        
    Return:
    array: list 
        stretched cubes
    '''
    array = [cubes[0],]
    values=[i for i in range(len(cubes[1]))]
    for k in range(1,len(cubes)):
        for word in itt.product(values, repeat=k):
            array= array+[cubes[k][word],]
    array = np.array(array)
    return array

def shuffle(a,b):#### a faster implementation is available if needed
    '''
    Inputs:
    a : list
        a word to be shuffled
    b : list
        b word to be shuffled
    
    Return:
    sh : list
        shuffle between a and b
    '''
    sh = []
    if len(a)==0:
        return [b,]
    if len(b)==0:
        return [a,]
    else:
        [sh.append([a[0],]+p) for p in shuffle(a[1:],b)]
        [sh.append([b[0],]+p) for p in shuffle(a,b[1:])]
    return sh


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



def OUL(d,m,Y_0,thetas,kappas,sigmas,Rho):   #Computes the matrix G^T associated to the drift operator
    '''                                      # See also implementation in Cuchiero, Svaluto-Ferro, Teichmann (2023), they are equivalent
    Inputs:
    d : int
    m : int 
    Y_0: list of initial values of the OU processes
    thetas: list of long-run means of the OUs
    kappas: list of reverting speeds of the OUs
    sigmas: list of volatilities of the OUs
    Rho: dxd np.array, correlation matrix of the OUs
    
    Returns:
    matrix: (d_m)x(d_m) np.array, G^T matrix associated to the operator
    '''
    
    dim=d+1
    matrix=np.zeros(int((dim**(m+1)-1)/(dim-1)))
    b_first=[kappas[i]*(thetas[i]-Y_0[i]) for i in range(d)]
    b_second=[-kappas[i] for i in range(d)]
    A_matrix=np.zeros((dim,dim))
    for i in range(1,dim):
        for j in range(1,dim):
            A_matrix[i,j]=(1/2)*Rho[i-1,j-1]*sigmas[i-1]*sigmas[j-1]
    
    for k in tqdm(range(1,m+1)):
        for word in itt.product([letters for letters in range(dim)],repeat=k):            
            cubes=cubes_zeros(d+1,m)
            if word[-1]==0:
                if k==1:
                    cubes[0]=1
                else:
                    cubes[k-1][tuple(word[:-1])] = 1
            else:#if the word ends with something different than zero/ i.e., diff. than time-component
                if k==1:
                    cubes[0]=cubes[0]+b_first[word[-1]-1]
                    cubes[k][word]=cubes[k][word]+b_second[word[-1]-1]  #fino qua corretto
                if k>1:
                    cubes[k-1][tuple(word[:-1])]=cubes[k-1][tuple(word[:-1])]+b_first[word[-1]-1]
                    
                    if k==2:
                        cubes[0]=cubes[0]+A_matrix[word[-1],word[-2]]                    
                    else:
                        cubes[k-2][tuple(word[:-2])]=cubes[k-2][tuple(word[:-2])]+A_matrix[word[-1],word[-2]]

                    for sh in shuffle(list(word)[:-1],[word[-1]]):
                        cubes[k][tuple(sh)]=cubes[k][tuple(sh)]+b_second[word[-1]-1]
                        
            newline = fromcubestoarray(cubes)
            matrix = np.vstack((matrix,newline))
    return matrix



def get_words(d,order_signature):
    '''
    Inputs: 
    d: int
    order_signature: int
    
    Returns:
    words: list of TUPLES (words) up to letter d and length order_signature
    '''
    words=[()]
    for k in range(1,order_signature+1):
        for word in itt.product([letters for letters in range(d+1)],repeat=k):
            words.append(word)
            
    return words

def get_words_list(d,order_signature):
    '''
    Inputs: 
    d: int
    order_signature: int
    
    Returns:
    words: list of LISTS (words) up to letter d and length order_signature
    '''
    words=[[]]
    for k in range(1,order_signature+1):
        for word in itt.product([letters for letters in range(d+1)],repeat=k):
            words.append(list(word))
            
    return words


def multidimensional_OU(X0,N,T,Rho,kappas,thetas,sigmas,flag_aug,flag_mat):
    '''
    Inputs: 
    X0: list of initial values of the OU processes
    N: int, number of grid points
    T: float/int final time
    Rho: dxd np.array, correlation matrix of the OUs
    kappas: list of reverting speeds of the OUs
    thetas: list of long-run means of the OUs
    sigmas: list of volatilities of the OUs
    flag_aug: if False, return (X_t) and not (t,X_t)
    flag_mat: if True, N is set to daily sampling (calendar days)
    
    Returns: 
    X: np.array (t,X_t)_{t\in[0,T]} 
    W: np.array (W_t) driving Bms
    
    '''
    
      ######################################################################################## Comment: if you 
      ####################### uncomment some chunks you will retrieve the underlying correlated brownian motions
      ####################### and you can switch to Euler-Maruyama instead that simulating directly from the solution of the SD
    if flag_mat==True:  #flag_mat is to sample up to a maturity time T>0 with daily sampling
        N=int(np.rint(T*365.25)) #flag_aug is to return (t,X_t) 

    dim=len(X0)
    A=np.zeros((dim,dim))
    A=np.matrix(A)
    for i in range(dim):
        for j in range(dim):
            A[i,j]=Rho[i,j]*sigmas[i]*sigmas[j]  
    C=cholesky(A)
    #C_1=cholesky(Rho)
    X = np.zeros([dim, int(N+1)])
    X[:,0]=X0
    #X_tilde[:,0]=X0
    T_vec, dt = np.linspace(0, T, N+1, retstep=True ) 
    print(dt)
    BMs=np.zeros([dim, int(N+1)])
    expy=np.array([np.exp(-kappas[j]*dt) for j in range(len(kappas))])
    Z = np.random.normal(0., 1., (dim,N+1))
    diffusion = np.matmul(C, Z)*np.sqrt(dt)
    thetas=np.array(thetas) ######################################################### up to here can be done outside the function to speed up
    for i in range(1,int(N+1)):
        #Z_i=Z[:,i]
        diffusion_i = diffusion[:,i].squeeze(1)  #these are diffusions including the BM
        #BMs[:,i]=np.matmul(C_1, Z_i)*np.sqrt(dt) #these are all BMs
        #X[:,i]=X[:,i-1]+kappas*(thetas-X[:,i-1])*dt+diffusion 
        X[:,i]=thetas+np.multiply((X[:,i-1]-thetas),expy)+diffusion_i 
    if flag_aug==False:
        return X,T_vec,np.cumsum(BMs,axis=1)
    else:
        return np.concatenate((np.expand_dims(T_vec,0),X),axis=0), np.cumsum(BMs,axis=1)

def sample_sig_OU_multi_minimal(N,T,X0,sigmas,kappas,thetas,Rho,order_signature,flag_mat):
    '''Inputs: same as the previous function'''
    
    augmented_OU, Bms =multidimensional_OU(X0,N,T,Rho,kappas,thetas,sigmas,True,flag_mat)
    augmented_OU_torch=torch.from_numpy(augmented_OU.transpose()).unsqueeze(0)
    sig=signatory.signature(augmented_OU_torch,order_signature,stream=True,basepoint=False,scalar_term=True)
    sig_arr=sig.squeeze().numpy()
    return sig_arr



def sample_sig_OU_multi_minimal_all_mat(N,maturities,X0,sigmas,kappas,thetas,Rho,order_signature,flag_mat):
    '''Inputs: in addition to the previous function
                  here it samples a full set of maturities
       maturities: np.array
    '''
    
    last_mat= maturities[-1]
    full_sig= sample_sig_OU_multi_minimal(N,last_mat+1/365.25,X0,sigmas,kappas,thetas,Rho,order_signature,flag_mat) 
    mat_in_days= [int(np.rint(m*365.25)) for m in maturities]
    get_model= [full_sig[m-1,:] for m in mat_in_days]
    get_model=np.array(get_model)
    return get_model

maturity=0.5
flag_mat=True
N=int(maturity*365.25) # put here the daily sampling
T=maturity # put here maturities
## first three a bit redundant 



 
if int(config[-1])==8:  
    d=4
    order_signature=3
    sigmas=[0.7,10,5,1]
    kappas=[0.1,25,10,0]
    X0=[1,0.08,2,0]
    thetas=[0.1,4,0.08,0]
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness/n=2/config2')
    Rho =np.load('Rho_d=4.npy')

if int(config[-1])==0:  ############ Only correlated Brownian motions
    d=3
    order_signature=3
    sigmas=[1,1,1]
    kappas=[0,0,0]
    X0=[0,0,0]
    thetas=[0,0,0]
    
    rng = np.random.default_rng(1267)
    Rho = random_correlation.rvs((2, 0.7, 0.3), random_state=rng)
    


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


keys_df=[str(word).replace(',','') if len(word)==1 else str(word).replace(' ','') for word in get_words(d,order_signature*2+1)]
words_as_lists=get_words_list(d,order_signature)
words_as_strings=[str(word).replace(" ", "") for word in get_words(d,order_signature)]


def e_tilde_multivariate(word,d):
    '''
    Inputs:
    word: list of ints
    d: int, "highest" letter considered
    
        
    Returns:
    new_words: list of lists, if word[-1]==0 add process component 
    new_words2: list of lists, only if word[-1]=!0, Ito-Stratonovich corrections
    
    '''
    
    the_components=np.array(range(d))+1
    new_words=[]
    for k in range(d):
        new_words.append(word.copy())
        new_words[k].append(the_components[k])
    if word[-1]==0:
        return new_words
    if word[-1]!=0: 
        new_words2=[]
        for j in range(d):
            new_words2.append(word.copy())
            new_words2[j][-1]=0
        return new_words, new_words2


def e_tilde_multivariate_part2(words_as_lists,d):
    '''
    Inputs:
    words_as_lists: list of lists of ints
    d: int components
    
    Returns: list of lists, tilde auxiliary output
    '''
    tilde=[list(e_tilde_multivariate(words_as_lists[k],d)) for k in np.array(range(len(words_as_lists)))[1:]]
    return tilde

tilde=e_tilde_multivariate_part2(words_as_lists,d)
tilde_copy=tilde.copy()
#print(tilde)

def list_to_string(my_list):
    '''
    Inputs: 
    my_list: list
    
    Returns: a string version of the list
    '''
    
    if len(my_list)==1:
        return str(tuple(my_list)).replace(",","")
    else:
        return str(tuple(my_list)).replace(" ","")

def from_tilde_to_string(tilde,d):
    '''    
    Inputs: 
    - The output of function e_tilde_multivariate_part2 (copied)
    d: int, components
    Returns: list of strings, the tilde transformation labels
    '''
    dimension_one_plus=d+1
    for k in range(len(tilde)):
        if k%dimension_one_plus==0:
            tilde[k]=[str(tuple(tilde[k][j])).replace(" ","") for j in range(len(tilde[k]))]
        else:
            tilde[k][0]=[list_to_string(element) for element in tilde[k][0]]
            tilde[k][1]=[list_to_string(element) for element in tilde[k][1]]
    
    return tilde


# IMPORTANT FOR TILDE TRANSFORMATION
new_tilde=from_tilde_to_string(tilde,d)


def get_cov_mat(sigmas,Rho,d):
    '''
    Inputs:
    sigmas: list of sigmas
    Rho_ex: dxd np.array, Rho correlation matrix
    d: int    
    '''   
    Rho_ex=np.zeros([d+1,d+1])
    Rho_ex[1:,1:]=Rho
    Cov=np.zeros([d+1,d+1])
    for j in range(1,d+1):
        for k in range(1,d+1):
            Cov[j,k]=Rho_ex[j,k]*(sigmas[j-1]*sigmas[k-1])
    return Cov



Cov=get_cov_mat(sigmas,Rho,d)
#print('Covariance matrix augmented for the time:\n',Cov)


def transform_df_multivariate(sig_df,new_tilde,order_signature,d,Cov): 
    '''
    Inputs:
    sig_df: pandas_df or pandas_series, output of function sample_sig_OU_multi()
    new_tilde: output of the tilde transformation
    order_signature: int, order of the signature
    d: int, components
    Cov: d+1xd+1 np.array, Covariance matrix (with additional row and columns for the time)
    
    Returns: 
    df_concat_new: pandas_df, signature after tilde transformation
    '''  
    words_as_lists=get_words_list(len(X0),order_signature)
    strings=[str(word).replace(',','') if len(word)==1 else str(word).replace(' ','') for word in get_words(len(X0),order_signature)]
    auxiliary_empty_lists = [[] for i in range(d)]
    
    for k in range(len(strings)):
        if k==0:
            for j in range(1,d+1):
                auxiliary_empty_lists[j-1].insert(0,sig_df[strings[j+1]])
        if ((k>0) and (words_as_lists[k][-1]==0)):
            for j in range(1,d+1):
                auxiliary_empty_lists[j-1].append(sig_df[new_tilde[k-1][j-1]])
        if ((k>0) and (words_as_lists[k][-1]!=0)):
            r=words_as_lists[k][-1]
            for j in range(1,d+1):
                auxiliary_empty_lists[j-1].append(sig_df[new_tilde[k-1][0][j-1]]-Cov[j,r]*0.5*sig_df[new_tilde[k-1][1][j-1]])
    helper_=[]
    for j in range(d):
        new_keys=[strings[k]+str('~Z_{}'.format(j+1)) for k in range(len(strings))]
        new_dict={key:series for key,series in zip(new_keys,auxiliary_empty_lists[j])}
        transformed_data_frame_W=pd.DataFrame(new_dict)
        helper_.append(transformed_data_frame_W)
    df_concat_new=pd.concat(helper_,axis=1)
    return df_concat_new


def get_tilde_keys(d,order_signature):
    strings=[str(word).replace(',','') if len(word)==1 else str(word).replace(' ','') for word in get_words(d,order_signature)]
    list_tilde_keys=[]
    for j in range(d):
        new_keys=np.array([strings[k]+str('~Z_{}'.format(j+1)) for k in range(len(strings))])
        list_tilde_keys.append(new_keys)
    return list(np.array(list_tilde_keys).flatten())


list_joint_maturities=np.sort(np.array(list(set(list(list_of_maturities_vix)+list(list_of_maturities_spx)))))
#indices=['Maturity T={}'.format(j) for j in range(1,max(len(list_of_maturities_spx),len(list_of_maturities_vix))+1)]
indices=['Maturity T={}'.format(j) for j in range(1,len(list_joint_maturities)+1)]
tilde_keys=get_tilde_keys(d,order_signature)
idx_Z_d=[important_key for important_key in tilde_keys if important_key[-3:]=='Z_'+str(d)]
keys_df_vix=[str(word).replace(',','') if len(word)==1 else str(word).replace(' ','') for word in get_words(d-1,order_signature*2)]


def remove_redundant_components(transformed_df,idx_Z_d,d,flag_not_dropped):
    '''
    Inputs:
    transformed_df: pandas DataFrame, tilde signature at maturity
    idx_Z_d: list of strings (labels of the last integral wrt Z_d)
    d: int, components
    flag_not_dropped: boolean, if True returns the index of the labels to be kept
    Returns:
    df: pandas DataFrame, tilde signature without the redundant components
    
    '''
    
    df=transformed_df[idx_Z_d]
    if flag_not_dropped==True:
        
        not_dropped=[] 
        for i,c in enumerate(df.columns):
            if str(d) in c[:-4]:
                df=df.drop(columns=[c])
            else:
                not_dropped.append(i)
        return df, not_dropped
    else:
        for i,c in enumerate(df.columns):
            if str(d) in c[:-4]:
                df=df.drop(columns=[c])
        return df 


def append_time(a):
    '''
    -Auxiliar function to append time to a list "a"
    '''
    aux=a.copy()
    aux.append(0)
    return [aux]

def shuffle_and_add_time(a,b):
    '''
    Inputs: #Only difference with the shuffle function is that we always append [0] (time-component) at the end
    a : array
        a word to be shuffled
    b : array
        b word to be shuffled
    
    Returns:
    sh : list
        shuffle between a and b
    '''
    sh = []
    if len(a)==0:
        return append_time(b)
    if len(b)==0:
        return append_time(a)
    else:
        [sh.append([a[0],]+p+[0]) for p in shuffle(a[1:],b)]
        [sh.append([b[0],]+p+[0]) for p in shuffle(a,b[1:])]
    return sh


def p_shuffle(order_signatrue,d):
    '''
    Inputs: 
    order_signature, d: int, int
    
    Returns:
    p, torch.tensor dimension d+1_n x d+1_n x d+1_2n 
    
    '''
    
    nbr_param_x2plus=number_of_parameters_gen(order_signature*2+1,d+1)
    dict_words_numbers=dict(zip(get_words(d,order_signature*2+1),[k for k in range(nbr_param_x2plus)]))
    p=[]
    wordz=get_words_list(d,order_signature)
    for word1 in tqdm(wordz):
        for word2 in wordz:
            sh=shuffle_and_add_time(word1,word2)
            p_components=np.zeros(nbr_param_x2plus)
            for shuffled in sh:
                p_components[dict_words_numbers[tuple(shuffled)]]=p_components[dict_words_numbers[tuple(shuffled)]]+1
            p.append(p_components)
    p=np.array(p)
    p=p.reshape((int(np.sqrt(p.shape[0])),int(np.sqrt(p.shape[0])),nbr_param_x2plus))
    return torch.tensor(p)

def p_shuffle_addtime(order_signatrue,d):
    '''
    Inputs: 
    order_signature, d: int, int
    
    Returns:
    p, torch.tensor dimension d_n x d_n x d_2n 
    
    '''
    
    nbr_param=number_of_parameters_gen(order_signature*2,d+1)
    dict_words_numbers=dict(zip(get_words(d,order_signature*2),[k for k in range(nbr_param)]))
    p=[]
    wordz=get_words_list(d,order_signature)
    for word1 in tqdm(wordz):
        for word2 in wordz:
            sh=shuffle(word1,word2)
            p_components=np.zeros(nbr_param)
            for shuffled in sh:
                p_components[dict_words_numbers[tuple(shuffled)]]=p_components[dict_words_numbers[tuple(shuffled)]]+1
            p.append(p_components)
    p=np.array(p)
    p=p.reshape((int(np.sqrt(p.shape[0])),int(np.sqrt(p.shape[0])),nbr_param))
    return torch.tensor(p)





def apply_along_axis(function, x, axis):
    torch_unbind=torch.unbind(x, dim=axis)
    aux=Parallel(n_jobs=-1)(delayed(function)(x_i) for x_i in tqdm(torch_unbind))
    return aux #torch.stack(aux,dim=axis)

def integration_torch(function,x,axis):
    return torch.trapz(torch.stack(apply_along_axis(function, x, axis),dim=axis),x)

if numerical_approx==True:
    
    G=OUL(d-1,2*order_signature,X0[:d-1],thetas[:d-1],kappas[:d-1],sigmas[:d-1],Rho[:d-1,:d-1]) #removing the BM component
    G_torch=torch.tensor(G)
nbr_param=number_of_parameters_gen(order_signature*2,d-1+1)
#print('Check if the dimension of G is correct:',G.shape == (nbr_param,nbr_param))
dict_words_numbers=dict(zip(get_words(d-1,order_signature*2),[k for k in range(nbr_param)]))
Delta=1/12

if numerical_approx==True:
    if np.max(np.abs(np.linalg.eig((G_torch*Delta).numpy())[0]))<1:
        print('Spectral radius:', np.max(np.abs(np.linalg.eig((G_torch*Delta).numpy())[0])))
        print('Spectral radius < 1, we can use Taylor expansion')
        len_expansion=100 #Computes the integral by expanding to len_expansion the definition of e^Gt
        taylor_exp=torch.eye(G_torch.shape[0])*Delta
        factorials=[np.math.factorial(k+1) for k in range(1,len_expansion)]
        for k in tqdm(range(1,len_expansion),desc='Taylor expansion'):
            k_th_term_taylor=Delta*torch.linalg.matrix_power(G_torch*Delta,k)*(factorials[k-1]**(-1))
            taylor_exp=taylor_exp+k_th_term_taylor
    else:
        nbins=50 #this can be pushed if nbr_parameters is not too high
        print('Spectral radius > 1, we can use Trapezoidal rule with N=',nbins)
        f2 = lambda t: torch.matrix_exp(t*G_torch)
        tv = np.linspace(0,Delta,nbins)
        torch_tv=torch.tensor(tv)
        for r in tqdm(range(1),desc='Trapezoidal rule'):
            exponential=integration_torch(f2,torch_tv,-1)


def sparse_dense_mul_vec(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:]]  # get values from relevant entries of dense matrix
    res=torch.matmul(v,dv)
    return res

def get_shuffled_integral_matrix(taylor_exp,order_signature,d):
    nbr_param=number_of_parameters_gen(order_signature*2,d+1)
    dict_words_numbers=dict(zip(get_words(d,order_signature*2),[k for k in range(nbr_param)]))
    wordz=get_words_list(d,order_signature)
    shuffled_integral_matrix=np.zeros((len(wordz),len(wordz),nbr_param))
    
    for i,word1 in enumerate(wordz):
        for j,word2 in enumerate(wordz):
            sh=shuffle(word1,word2)
            p_components=np.zeros(nbr_param)
            for shuffled in sh:
                p_components[dict_words_numbers[tuple(shuffled)]]=p_components[dict_words_numbers[tuple(shuffled)]]+1
                shuffled_integral_matrix[i,j,:]=sparse_dense_mul_vec(torch.tensor(p_components).to_sparse(),taylor_exp)
    return torch.tensor(shuffled_integral_matrix)

# This commented part is to check convergence of the the trapezoidal towards the Taylor and viceversa
# =============================================================================
# sh1=get_shuffled_integral_matrix(taylor_exp,order_signature,d-1)
# sh2=get_shuffled_integral_matrix(exponential,order_signature,d-1)
# 
# error_matrix=(torch.matmul(G_torch,f2(Delta))-G_torch)*(-(Delta**3)/(nbins**2))
# print('1000 highest Error analysis matrix:', torch.sort(error_matrix.flatten()).values[-1000:])
# 
# 
# print('Comparison shuffled integral matrix', torch.sort(torch.abs(sh1-sh2).flatten()).values[-1000:])
# 
# =============================================================================

if numerical_approx==True:
        
    if np.max(np.abs(np.linalg.eig((G_torch*Delta).numpy())[0]))<1:
        shuffled_integral_matrix=get_shuffled_integral_matrix(taylor_exp,order_signature,d-1)
    else:
        shuffled_integral_matrix=get_shuffled_integral_matrix(exponential,order_signature,d-1)
        
    
    print('Shuffled_integral_matrix shape:',shuffled_integral_matrix.shape)
    
    print('Dimension G:', f2(Delta).shape)







def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args),repeat=2), itt.product(*args,repeat=2))



##################### COMPUTATION OF THE Q^(0,CV)   FOR THE SPX

if flag_gatheral==True:
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness_Gatheral2/n='+str(order_signature)+'/'+config)
else:
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness/n='+str(order_signature)+'/'+config)


if available_Gplus==True:
    dim=len(get_words(d-1,order_signature*2+1))
    G_torch_plus=torch.tensor(np.load('G_plus(('+str(dim)+', '+str(dim)+')).npy'))
    print('Loaded G shape:',G_torch_plus.shape)
else:
    G_plus=OUL(d-1,2*order_signature+1,X0,thetas,kappas,sigmas,Rho) 
    G_torch_plus=torch.tensor(G_plus)
    shape_G_plus=G_torch_plus.shape
    np.save(f'G_plus({shape_G_plus}).npy',G_torch_plus.numpy())

f3 = lambda t: torch.matrix_exp(t*G_torch_plus)

wordz=get_words_list(d-1,order_signature)
dict_words_numbers=dict(zip(get_words(d-1,order_signature*2+1),[k for k in range(len(get_words(d-1,order_signature*2+1)))]))
nbr_param_x2plus=len(get_words(d-1,order_signature*2+1))


######## VIX



def get_shuffled_integral_matrix_withtime(matrix,order_signature,d):
    nbr_param=number_of_parameters_gen(order_signature*2+1,d+1)
    dict_words_numbers=dict(zip(get_words(d,order_signature*2+1),[k for k in range(nbr_param)]))
    wordz=get_words_list(d,order_signature)
    print(wordz)
    shuffled_integral_matrix=np.zeros((len(wordz),len(wordz),nbr_param))
    print(nbr_param)
    for i,word1 in enumerate(wordz):
        for j,word2 in enumerate(wordz):
            sh=shuffle_and_add_time(word1,word2)
            p_components=np.zeros(nbr_param)
            for shuffled in sh:
                p_components[dict_words_numbers[tuple(shuffled)]]=p_components[dict_words_numbers[tuple(shuffled)]]+1
                shuffled_integral_matrix[i,j,:]=sparse_dense_mul_vec(torch.tensor(p_components).to_sparse(),matrix)
    return torch.tensor(shuffled_integral_matrix)


print('list joint maturities:',list_joint_maturities)

sig_trivial=torch.zeros([nbr_param_x2plus]).type(torch.DoubleTensor)
sig_trivial[0]=1

if flag_save_control_variates==True:
    exps=[f3(mat+Delta)-f3(mat) for mat in tqdm(list_of_maturities_vix, desc='exp vix')]
    exps2=[f3(mat) for mat in tqdm(list_joint_maturities, desc='exp joint')]
else:
    print('No CV')
    
matrix_exp_id=f3(Delta)-torch.eye(nbr_param_x2plus)


print('Checkpoint before CV sampling')

if numerical_approx==False:
    if flag_save_control_variates==True:
        Q_cv=torch.stack([torch.matmul(get_shuffled_integral_matrix_withtime(matrix,order_signature,d-1),sig_trivial) for matrix in tqdm(exps,desc='Q_cv')])
        Q0_cv=torch.stack([torch.matmul(get_shuffled_integral_matrix_withtime(matrix,order_signature,d-1),sig_trivial) for matrix in tqdm(exps2,desc='Q_cv0')])
        print('Shape Q0:cv', Q0_cv.shape)
        print('Shape Q:cv', Q_cv.shape)
       
        shape_Q0_cv=Q0_cv.shape
        shape_Q_cv=Q_cv.shape
   
        np.save(f'CV_exact_SPX({shape_Q0_cv}).npy',Q0_cv.numpy())
        np.save(f'CV_exact_VIX({shape_Q_cv}).npy',Q_cv.numpy())
        

        shuffled_integral_matrix=get_shuffled_integral_matrix_withtime(matrix_exp_id,order_signature,d-1)
    else:
        shuffled_integral_matrix=get_shuffled_integral_matrix_withtime(matrix_exp_id,order_signature,d-1)
        




print('Checkpoint after CV sampling')


# dimension Q0_cv : (nbr_maturities_joint, nbr_parameters_to_calibrate, nbr_parameters_to_calibrate)
# type Q0_cv: torch.tensor
# dimension Q_cv : (nbr_maturities_vix, nbr_parameters_to_calibrate, nbr_parameters_to_calibrate)
# type Q0_cv: torch.tensor


if flag_gatheral==True:
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness_Gatheral2/n='+str(order_signature)+'/'+config)
else:
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness/n='+str(order_signature)+'/'+config)



def chol_n_transpose(q):
    '''
    Inputs:
    q: np.array, positive semidefinite symmetric matrix
    Returns:
    U: np.array, upper triangular matrix of Cholesky decomposition
    '''
    return np.linalg.cholesky(q).transpose()

def filter_positive_semidef(Q,idx_mat):
    nbr_params=Q.shape[-1]
    Q=Q[idx_mat,:,:,:].numpy()
    check_eigenvalues=np.sum(np.linalg.eigvalsh(Q)>0,axis=1)
    bad_indices=np.where(check_eigenvalues<nbr_params)
    print('Idx Maturity:'+str(idx_mat)+', Number of bad indices: {}'.format(len(bad_indices[0])))
    Q=np.delete(Q,bad_indices,0)
    return Q
          
    
def cholesky_dec(Q,idx_mat):
    '''
    Apply Cholesky decomposition to matrix Q, by maturity (idx_mat=int >0)
    Returns: list of Cholesky decompositions
    '''
    
    Q=filter_positive_semidef(Q,idx_mat)
    adjusted_mc_number=Q.shape[0]  
    list_=np.array(Parallel(n_jobs=-1)(delayed(chol_n_transpose)(Q[j,:,:]) for j in tqdm(range(adjusted_mc_number),desc='Chol')))
    return list_


print('List of joint maturities:',list_joint_maturities)


wordz=get_words_list(d-1,order_signature)
dict_words_numbers=dict(zip(get_words(d-1,order_signature*2+1),[k for k in range(len(get_words(d-1,order_signature*2+1)))]))


def sample_tilde_df_andQ0_multimaturities(N,maturities,mat_spx,mat_vix,X0,sigmas,kappas,thetas,Rho,order_signature,flag_mat,indices,new_tilde,keys_df,keys_df_vix,Cov,d,idx_Z_d,shuffled_integral_matrix,dict_words_numbers,wordz):
    
   
    idx_vix=np.nonzero(mat_vix[:, None] == maturities)[1]
    
    
    # Raw sig
    sig=sample_sig_OU_multi_minimal_all_mat(N,maturities,X0,sigmas,kappas,thetas,Rho,2*order_signature+1,flag_mat)
    sig_df_spx=pd.DataFrame(data=sig,index=indices , columns=keys_df)
    sig_df_vix=pd.DataFrame(data=sig[idx_vix],index=indices[:len(mat_vix)], columns=keys_df)
    
        
    # For spx
    transformed_sig_df=transform_df_multivariate(sig_df_spx,new_tilde,order_signature,d,Cov)
    transformed_sig_df, not_dropped=remove_redundant_components(transformed_sig_df,idx_Z_d,d,True)
    transformed_sig_df= torch.tensor(np.array(transformed_sig_df))
    
    keys_df_vix2=[str(word).replace(',','') if len(word)==1 else str(word).replace(' ','') for word in get_words(d-1,order_signature*2+1)]
    sig_df_spx=sig_df_spx.loc[:,keys_df_vix2]

    
    nbr_param_x2plus=sig_df_spx.shape[1] 
    Q0_all=np.zeros((len(maturities),len(not_dropped),len(not_dropped)))
    sig_df_spx=torch.tensor(np.transpose(np.array(sig_df_spx)))
    
      
    for idx,word1 in enumerated_product(wordz):
        sh=shuffle_and_add_time(word1[0],word1[1])
        p_components=np.zeros(nbr_param_x2plus)
        for shuffled in sh:
            p_components[dict_words_numbers[tuple(shuffled)]]=p_components[dict_words_numbers[tuple(shuffled)]]+1
            Q0_all[:,idx[0],idx[1]]=torch.matmul(torch.tensor(p_components),sig_df_spx)
    
    Q0_all=torch.tensor(Q0_all)
    # For vix
    Q=torch.matmul(shuffled_integral_matrix,torch.transpose(torch.tensor(sig_df_vix[keys_df_vix2].values),0,1))
    
    
    return [transformed_sig_df, Q0_all,Q]





# CHANGE HERE THE DIRECTORY WHERE YOU WOULD LIKE TO STORE THE RANDOMNESS 


if flag_gatheral==True:
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness_Gatheral2/n='+str(order_signature)+'/'+config)
else:
    os.chdir(r'/scratch.global/ag_cu/Codes_Guido/Randomness/n='+str(order_signature)+'/'+config)




rounds=8
MC_nbr=10000
print('Checkpoint before sampling')

if flag_missing_last==True:
    for u in tqdm(range(1),desc='Slice samples'):
        
        
        random_components = Parallel(n_jobs=-1)(delayed(sample_tilde_df_andQ0_multimaturities)(N,list_joint_maturities,list_of_maturities_spx,list_of_maturities_vix,X0,sigmas,kappas,thetas,Rho,order_signature,flag_mat,indices,new_tilde,keys_df,keys_df_vix,Cov,d,idx_Z_d,shuffled_integral_matrix,dict_words_numbers,wordz) for j in tqdm(range(MC_nbr)))
        random_components = [item for sublist in random_components for item in sublist]
        

        E_sig_B=torch.transpose(torch.stack(random_components[0::3]),0,1)
        Q0=torch.transpose(torch.stack(random_components[1::3]),0,1)
        Q=torch.transpose(torch.transpose(torch.stack(random_components[2::3]),0,3),1,3)

         
        if flag_chol_vix==True:
            L=np.array([cholesky_dec(Q,k) for k in tqdm(range(Q.shape[0]),desc='By Maturity')])
            L_torch=torch.tensor(L)
            np.save(f'L({order_signature},{d},7).npy',L)
        else:
            np.save(f'Q({order_signature},{d},7).npy',Q.numpy())
        
        
        np.save(f'E_sig_B({order_signature},{d},7).npy',E_sig_B.numpy())
        np.save(f'Q_0({order_signature},{d},7).npy',Q0.numpy())
        
    
else:
    
    print('Total number of Monte Carlo Samples will be :',int(MC_nbr*rounds))
    
    for u in tqdm(range(rounds),desc='Slice samples'):
        
        
        random_components = Parallel(n_jobs=-1)(delayed(sample_tilde_df_andQ0_multimaturities)(N,list_joint_maturities,list_of_maturities_spx,list_of_maturities_vix,X0,sigmas,kappas,thetas,Rho,order_signature,flag_mat,indices,new_tilde,keys_df,keys_df_vix,Cov,d,idx_Z_d,shuffled_integral_matrix,dict_words_numbers,wordz) for j in tqdm(range(MC_nbr)))
        random_components = [item for sublist in random_components for item in sublist]
        
    
        E_sig_B=torch.transpose(torch.stack(random_components[0::3]),0,1)
        Q0=torch.transpose(torch.stack(random_components[1::3]),0,1)
        Q=torch.transpose(torch.transpose(torch.stack(random_components[2::3]),0,3),1,3)
    
         
        
        if flag_chol_vix==True:
            L=np.array([cholesky_dec(Q,k) for k in tqdm(range(Q.shape[0]),desc='By Maturity')])
            L_torch=torch.tensor(L)
            np.save(f'L({order_signature},{d},{u}).npy',L)
        else:
            np.save(f'Q({order_signature},{d},{u}).npy',Q.numpy())
        
        
        
        
        np.save(f'E_sig_B({order_signature},{d},{u}).npy',E_sig_B.numpy())
        np.save(f'Q_0({order_signature},{d},{u}).npy',Q0.numpy())
       
        np.save('Rho_d='+str(d)+'.npy',Rho)
            
        
        
