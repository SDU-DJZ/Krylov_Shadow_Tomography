"""
Figure for application 2: bound entangled states

rho_NK=lambda*P_NK+lambda*(Q+_NK+Q-_NK)/2

H=sum of local Pauli-Z matrices/2
"""
import os
import math
import numpy as np
import json
import multiprocessing
import matplotlib.pyplot as plt
import my_functions as mf
import data_acquisition as da
import prediction_QFI as pr

"""
Data generated for subfigure a
"""
def lamb(N, K):
    if K>=math.floor(N/2):
        print("K must be less than the floor of N divided by 2.")
    
    lamb_inv=0
    for i in range(K+1):
        lamb_inv+=math.comb(N, i)
    return 1/lamb_inv


def QFI(N, K):
    if K>=math.floor(N/2):
        print("K must be less than the floor of N divided by 2.")
        
    term1=lamb(N, K)
    term2=0 
    for j in range(K):
        term2+=(N-2*j)**2*math.comb(N,j)
    
    FQ=term1*term2
    return FQ


def B_pow(N, K, order):
    lamb_cal=lamb(N, K)
    QFI_cal=QFI(N, K)
    B=(1-(1-lamb_cal)**(order+1))*QFI_cal    
    return B


def B_leg(N, K):
    lamb_cal=lamb(N, K)
    f = lamb_cal   
    # 使用 np.any() 或 np.all() 检查条件
    B = np.where(f > 1/2, N**2 * (1 - 2*f)**2, 0)      
    return B


# Number of qubits
N = 12
K_Index=range(1, math.floor(N/2))

#Initialization
QFI_List=[]
B_leg_List=[]
B_sub_List=[]
B1_pow_List=[]
B2_pow_List=[]
B3_pow_List=[]


#Reletive error
for K in K_Index:
    QFI_List.append(QFI(N, K))
    B_leg_List.append(B_leg(N, K))
    B_sub_List.append(B_pow(N, K, 0))
    B1_pow_List.append(B_pow(N, K, 1))
    B2_pow_List.append(B_pow(N, K, 2))
    B3_pow_List.append(B_pow(N, K, 3))
    
    
error_leg=np.abs(np.array(QFI_List)-np.array(B_leg_List))/np.array(QFI_List)
error_sub=np.abs(np.array(QFI_List)-np.array(B_sub_List))/np.array(QFI_List)
error1_pow=np.abs(np.array(QFI_List)-np.array(B1_pow_List))/np.array(QFI_List)
error2_pow=np.abs(np.array(QFI_List)-np.array(B2_pow_List))/np.array(QFI_List)
error3_pow=np.abs(np.array(QFI_List)-np.array(B3_pow_List))/np.array(QFI_List)
error1_our=np.abs(np.array(QFI_List)-np.array(QFI_List))/np.array(QFI_List)


"""
Data loaded for subfigure b: average_errors as a function of p_list
"""
k_list=np.arange(1, 4)
with open('numerical_results_app2_k.json', 'r', encoding='utf-8') as file:
    data_app2_k_all = json.load(file)
    data_app2_k=data_app2_k_all["data9"]
    average_errors=data_app2_k["average_errors"]

# system_size=6   M=3200000


"""
Data loaded for subfigure c: M as a function of N
"""
N_list=np.arange(2, 7)

with open('numerical_results_app2_M.json', 'r', encoding='utf-8') as file:
    data_app2_M_all = json.load(file)

M_list=[]
for round_label in range(1,6):
    M_value=data_app2_M_all[f"data{round_label}"]["M, B1, B2, and L are"][0]
    M_list.append(M_value)



"""
Plots
"""
# fig=plt.figure(figsize=(10,6))

ax1=plt.subplot(121)

ax1.plot(K_Index, error_leg, 'x--g', label=r'$B^{(\mathsf{Leg})}$')
ax1.plot(K_Index, error_sub, '<--b', label=r'$B^{(\mathsf{Sub})}$')
ax1.plot(K_Index, error1_pow, '*--r', label=r'$B_1^{(\mathsf{Tay})}$')
ax1.plot(K_Index, error2_pow, '*--y', label=r'$B_2^{(\mathsf{Tay})}$')
ax1.plot(K_Index, error3_pow, '*--c', label=r'$B_3^{(\mathsf{Tay})}$')
ax1.plot(K_Index, error1_our, 'D--k', label=r'$B_1^{(\mathsf{Kry})}$')

ax1.set_xlabel(r'$k$',fontsize=14)
ax1.set_ylabel(r'$\mathcal{E}$',fontsize=14)


plt.legend()
ax1.text(-0.3, 1, '(a)', fontsize=14, transform=ax1.transAxes)


ax2=plt.subplot(222)
ax2.plot(k_list, average_errors, 'D--k')

ax2.set_xlabel(r'$k$',fontsize=14)
ax2.set_ylabel(r'$\hat{\mathcal{E}}$',fontsize=14)
ax2.set_ylim(-0.1, 1)
ax2.set_xticks([1, 2, 3])

ax2.text(-0.3, 1, '(b)', fontsize=14, transform=ax2.transAxes)

ax3=plt.subplot(224)
ax3.plot(N_list, M_list, 'D--k')
ax3.set_yscale('log')

ax3.set_xlabel(r'$N$',fontsize=14)
ax3.set_ylabel(r'$M$',fontsize=14)

ax3.text(-0.3, 1, '(c)', fontsize=14, transform=ax3.transAxes)

plt.tight_layout()

plt.savefig('fig_bound_entangled_states_comp.pdf', dpi=600, bbox_inches='tight', pad_inches=0.01)

plt.show()


