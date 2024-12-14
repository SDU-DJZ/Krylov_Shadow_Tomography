"""
Store all the functions needed
"""


import numpy as np
import qutip as qt
import itertools
import math



"""
Generate the Hamiltonian for an N-qubit system.
"""
def generate_hamiltonian(N):
    # Initialize Hamiltonian to zero
    dims=[[2]*N]
    hamiltonian = qt.qzero(dims)
    
    # Loop through each term, which is a Kronecker product of Pauli-Z and identity matrices
    for i in range(N):
        # Create the Pauli-Z matrix for the i-th term
        pauli_z_i = qt.sigmaz()
        
        # Create the identity matrix for other terms
        identity_i = qt.identity(2)
        
        # Compute the Kronecker product for the i-th term
        term_i =(1/2)*qt.tensor([identity_i] * i + [pauli_z_i] + [identity_i] * (N - i - 1))
        
        # Add the current term to the total Hamiltonian
        hamiltonian += term_i
    
    return hamiltonian


"""
Generate random density matrices
"""
def generate_random_dm(N):
    rho = qt.rand_dm(2**N)
    new_dims = [[2]*N, [2]*N]

    # Resize rho to the new dimensions
    rho_resized = qt.Qobj(rho.data, dims=new_dims)
    return rho_resized

"""
Generate the N-qubit state of interest: rho=(1-p)|GHZ><GHZ|+pI/d
"""
def generate_GHZ_type_dm(p, N):
    # system_size: the number of qubits
    b0=qt.basis(2,0)
    b1=qt.basis(2,1)
    GHZ=(qt.tensor([b0]*N)+qt.tensor([b1]*N)).unit()
    I2=qt.identity(2)
    Id=qt.tensor([I2]*N)
    rho=(1-p)*GHZ*GHZ.dag()+p*Id/2**N
    return rho


"""
Generate the bound entangled state
"""
def lamb(N, K):
    if K>math.floor(N/2):
        print("K must be less than the floor of N divided by 2.")
    
    lamb_inv=0
    for i in range(K+1):
        lamb_inv+=math.comb(N, i)
    return 1/lamb_inv

# Generate phi_i
def phi(N, comb, sign):    
    b0=qt.basis(2,0)
    b1=qt.basis(2,1)
    index_set=list(comb)
    # Initialize basic_state
    if 0 in index_set:
        basic_state0=b1
        basic_state1=b0
    else:
        basic_state0=b0
        basic_state1=b1
        
    for i in range(1, N):
        if i in index_set:
            basic_state0=qt.tensor(basic_state0, b1)
            basic_state1=qt.tensor(basic_state1, b0)
        else:
            basic_state0=qt.tensor(basic_state0, b0)
            basic_state1=qt.tensor(basic_state1, b1)
    
    if sign=='+':
        phi=(basic_state0+basic_state1).unit()
    else:
        phi=(basic_state0-basic_state1).unit()
    return phi
        
# Sum phi for a fixed HW j
def sum_phi(N, j, sign):
    qubit_index=list(range(N))
    all_comb=itertools.combinations(qubit_index, j)
    
    # Initialize sum_phi
    sum_phi=qt.Qobj(np.zeros((2**N, 2**N)), dims=[[2]*N, [2]*N])
    for comb in all_comb:
        comb_list=list(comb)
        phi_single=phi(N, comb_list, sign)
        sum_phi+=phi_single*phi_single.dag()
    return sum_phi
        
        
def P(N, k):
    if k==0:
        term=0 
    else:
        term=sum_phi(N, 0, '+')
        for j in range(1, k):
            term+=sum_phi(N, j, '+')
    return term
    
def generate_bound_entangled_state(N, k):
    P_term=P(N, k)
    Q0_term=sum_phi(N, k, '+')
    Q1_term=sum_phi(N, k, '-')
    lamb_term=lamb(N, k)
    rho=lamb_term*P_term+0.5*lamb_term*(Q0_term+Q1_term)
    return rho 
    

"""
Caculating QFI
"""
def quantum_fisher_information(rho, H):
    # Get eigenvalues and eigenstates of the density matrix rho
    evals, evecs = rho.eigenstates()

    # Initialize QFI
    QFI = 0

    # Calculate the QFI using the given formula
    for k in range(len(evals)):
        for l in range(len(evals)):
            if k != l:
                p_k = evals[k]
                p_l = evals[l]
                
                if p_k + p_l > 1e-10:  # Avoid division by zero
                    # Calculate the inner product of commutator and its adjoint
                    inner_product = (evecs[k].dag() * H * evecs[l])
                    inner_product = inner_product * inner_product.conjugate()
                    
                    QFI += 2 * (p_k - p_l)**2 / (p_k + p_l) * inner_product

    return QFI.real

    
"""
Caculating lower bounds proposed in PRL 127, 260501(2021)

order=0,1,2,3... denotes the order of the bound
"""   
def lower_bound(rho, H, order, N):
    #Initialize Bound
    bound=0
    
    rho_full=rho.full()
    H_full=H.full()
    I_full=qt.identity(2**N).full()
    S_full=qt.swap(2**N,2**N).full()
    
    for i in range(order+1):
        term=2*np.trace(np.linalg.matrix_power(np.kron(rho_full,I_full)-np.kron(I_full,rho_full),2)@np.linalg.matrix_power(np.kron(I_full,I_full)-np.kron(rho_full,I_full)-np.kron(I_full,rho_full), i)@S_full@np.kron(H_full,H_full))
        bound+=term
    
    return np.real(bound)
    
    
"""
Caculating Krylov QFI
"""


def T(rho, H,  order, N):
    rho_full=rho.full()
    H_full=H.full()
    H_vec=np.reshape(H_full,(4**N,1))
    I_full=np.identity(2**N)
    
    R=(1/2)*(np.kron(rho_full, I_full)+np.kron(I_full, rho_full.T))    
    kron_diff = np.kron(rho_full, I_full) - np.kron(I_full, rho_full.T)
    
    T_result = np.conj(H_vec).T @ kron_diff @ np.linalg.matrix_power(R, order) @ kron_diff @ H_vec
    
    return np.real(T_result)
    
    
def Hankel_matrix(rho, H, order, N):
    h_matrix=np.zeros((order,order))
    for k in range(order):
        for l in range(order):
            h_matrix[k,l]=T(rho,H,k+l+1,N)
    return h_matrix
    
def b_vec(rho, H, order, N):
    b=np.zeros(order)
    for k in range(order):
        b[k]=T(rho,H,k,N)
    return b
        
def Krylov_QFI(rho, H, order, N):
    Hank=Hankel_matrix(rho,H,order,N)
    b=b_vec(rho,H,order,N)
    Krylov_QFI=np.conj(b).T@np.linalg.inv(Hank)@b
    return np.real(Krylov_QFI)
    
    






    



