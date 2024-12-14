"""
Figure for application 1: pseudo-pure states

rho= (1-p) |GHZ><GHZ|+ p I/d

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

def QFI(N, p):
    FQ=N**2*(1-p)**2/(1-p+2*p/2**N)
    return FQ

def B_leg(N, p):
    f = 1- p + p/ 2**N    
    # 使用 np.any() 或 np.all() 检查条件
    B = np.where(f > 1/2, N**2 * (1 - 2*f)**2, 0)    
    return B

def B_pow(N, p, order):
    B=(1-(1-2/2**N)**(order+1)*p**(order+1))*N**2*(1-p)**2/(1-p+2*p/2**N)
    return B


# Number of qubits
N = 12
prob=np.linspace(0, 0.99, 100)


#Reletive error
quant_FI=QFI(N, prob)
bound_leg=B_leg(N, prob)
bound_sub=B_pow(N, prob, 0)

bound0_pow=B_pow(N, prob, 0)
bound1_pow=B_pow(N, prob, 1)
bound2_pow=B_pow(N, prob, 2)
bound3_pow=B_pow(N, prob, 3)
bound1_our=QFI(N, prob)

error_leg=np.abs(quant_FI-bound_leg)/quant_FI
error_sub=np.abs(quant_FI-bound_sub)/quant_FI

error0_pow=np.abs(quant_FI-bound0_pow)/quant_FI
error1_pow=np.abs(quant_FI-bound1_pow)/quant_FI
error2_pow=np.abs(quant_FI-bound2_pow)/quant_FI
error3_pow=np.abs(quant_FI-bound3_pow)/quant_FI
error1_our=np.abs(quant_FI-bound1_our)/quant_FI


"""
Data loaded for subfigure b: average_errors as a function of p_list
"""

p_list=np.linspace(0, 0.9, 5)
with open('numerical_results_app1_p.json', 'r', encoding='utf-8') as file:
    data_app1_p_all = json.load(file)
    data_app1_p=data_app1_p_all["data3"]
    average_errors=data_app1_p["average_errors"]

# system_size=6   M=480000


"""
Data loaded for subfigure c: M as a function of N
"""

N_list=np.arange(2, 8)

with open('numerical_results_app1_M.json', 'r', encoding='utf-8') as file:
    data_app1_M_all = json.load(file)

M_list=[]
for round_label in range(1,7):
    M_value=data_app1_M_all[f"data{round_label}"]["M, B1, B2, and L are"][0]
    M_list.append(M_value)



"""
Plots
"""

# fig=plt.figure(figsize=(10,6))

ax1=plt.subplot(121)
 

ax1.plot(prob, error_leg, 'g', label=r'$B^{(\mathsf{Leg})}$')
ax1.plot(prob, error_sub, 'b', label=r'$B^{(\mathsf{Sub})}$')
ax1.plot(prob, error1_pow, 'r', label=r'$B_1^{(\mathsf{Tay})}$')
ax1.plot(prob, error2_pow, 'y', label=r'$B_2^{(\mathsf{Tay})}$')
ax1.plot(prob, error3_pow, 'c', label=r'$B_3^{(\mathsf{Tay})}$')
ax1.plot(prob, error1_our, 'k', label=r'$B_1^{(\mathsf{Kry})}$')


ax1.set_xlabel(r'$p$',fontsize=14)
ax1.set_ylabel(r'$\mathcal{E}$',fontsize=14)

plt.legend()
ax1.text(-0.3, 1, '(a)', fontsize=14, transform=ax1.transAxes)


ax2=plt.subplot(222)
ax2.plot(p_list, average_errors, 'D--k')

ax2.set_xlabel(r'$p$',fontsize=14)
ax2.set_ylabel(r'$\hat{\mathcal{E}}$',fontsize=14)
ax2.set_ylim(-0.1, 1)
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8])

ax2.text(-0.3, 1, '(b)', fontsize=14, transform=ax2.transAxes)

ax3=plt.subplot(224)
ax3.plot(N_list, M_list, 'D--k')
ax3.set_yscale('log')

ax3.set_xlabel(r'$N$',fontsize=14)
ax3.set_ylabel(r'$M$',fontsize=14)
ax3.set_xticks([2, 3, 4, 5, 6, 7])

ax3.text(-0.3, 1, '(c)', fontsize=14, transform=ax3.transAxes)

plt.tight_layout()

plt.savefig('fig_pseudo_pure_states_comp.pdf', dpi=600, bbox_inches='tight', pad_inches=0)

plt.show()


