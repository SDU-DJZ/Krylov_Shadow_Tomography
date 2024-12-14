import numpy as np
import qutip as qt
import itertools


#Warning: rho-->rho.full()   H-->H.full()


# define a dictionary for Pauli matrices
dict_pauli_operators = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}

# define a dictionary for "+-"

dict_outcomes = {"+": 1, "-": -1}


"""
Construct a classical shadow from one measurement datum
"""

def classical_shadow(measurement_datum):
    N =np.int_(len(measurement_datum)/2)      

    reduced_shadow_list = [
        (1 / 2) * qt.identity(2)
        + (3 / 2)
        * dict_outcomes[measurement_datum[i + N]]
        * dict_pauli_operators[measurement_datum[i]]
        for i in range(N)
    ]

    shadow = qt.tensor(reduced_shadow_list).full()
    return shadow


"""
Generate batch shadows from measurement data
"""
def batch_shadows(measurement_data, B, L):
    #Initialize batch shadows
    batch_shadows=[]
    #Split all the measurement data into B batches
    batches = np.array_split(measurement_data, B)
    
    for b in range(B):
        shadow_list=[classical_shadow(batches[b][l]) for l in range(L)]
        
        batch_shadows.append(np.mean(shadow_list, axis=0))
    return batch_shadows
    
    
"""
Define T0_hat and T1_hat
"""
# define T0_hat
def T0_hat_term(shadow1, shadow2, H):     
    commute1 = shadow1@H-H@shadow1
    commute2 = shadow2@H-H@shadow2    
    term = -np.trace(commute1@commute2)
    return np.real(term)


def T0_hat(batch_shadows, B1, B2, H):
    #Split batch_shadows for performing MoM estimation
    batch_shadows_split=np.array_split(batch_shadows, B1)
    T0_hat_list=[]
    I_permutations=list(itertools.permutations(range(B2), 2))
    num = len(I_permutations)
    for b1 in range(B1):
        sum_term = 0
        for j in range(num):
            shadow1 = batch_shadows_split[b1][I_permutations[j][0]]
            shadow2 = batch_shadows_split[b1][I_permutations[j][1]]
            term = T0_hat_term(shadow1, shadow2, H)
            sum_term += term
        T0_hat_list.append(sum_term / num)

    T0_hat = np.median(T0_hat_list)
    return np.real(T0_hat)


# define T1_hat
def T1_hat_term(shadow1, shadow2, shadow3, H):
    commute1 = shadow1@H-H@shadow1
    commute2 = shadow2@H-H@shadow2

    term = -(1/2)*np.trace(commute1@(shadow3@commute2+commute2@shadow3))
    return np.real(term)


def T1_hat(batch_shadows, B1, B2, H):
    #Split batch_shadows for performing MoM estimation
    batch_shadows_split=np.array_split(batch_shadows, B1)
    T1_hat_list=[]
    I_permutations=list(itertools.permutations(range(B2), 3))
    num = len(I_permutations)
        
    for b1 in range(B1):
        sum_term = 0
        for j in range(num):
            shadow1 = batch_shadows_split[b1][I_permutations[j][0]]
            shadow2 = batch_shadows_split[b1][I_permutations[j][1]]
            shadow3 = batch_shadows_split[b1][I_permutations[j][2]]
            term = T1_hat_term(shadow1, shadow2, shadow3, H)
            sum_term += term
        T1_hat_list.append(sum_term / num)

    T1_hat = np.median(T1_hat_list)
    return np.real(T1_hat)


"""
Predict the 1st-order Krylov QFI
"""
def F1_hat(batch_shadows, B1, B2, H):
    F1_est=T0_hat(batch_shadows, B1, B2, H) ** 2 / T1_hat(batch_shadows, B1, B2, H)
    return F1_est




