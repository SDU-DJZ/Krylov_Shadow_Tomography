"""
Created on Wed Jun 19 16:50:06 2024

@author: Da-Jian Zhang

"""

import numpy as np
import qutip as qt
import itertools


# define a dictionary for Pauli matrices 
dict_pauli_operators = {
    'X': qt.sigmax(),
    'Y': qt.sigmay(),
    'Z': qt.sigmaz()
}

# define a dictionary for "+-"

dict_outcomes={'+': 1, '-': -1}


"""
Randomly choosing local Pauli measurements 
"""
def randomized_measurement_procedure(num_total_measurements, system_size):
    
    # num_total_measurements: int for the total number of measurement rounds
    # system_size: int for how many qubits in the quantum system
    
    measurement_procedure = []
    for t in range(num_total_measurements):
        single_round_measurement =''.join([np.random.choice(["X", "Y", "Z"]) for i in range(system_size)])
        measurement_procedure.append(single_round_measurement)
    return measurement_procedure

"""
Numerically simulating the physical experiment
"""
# Generating a list of all the possible measurement outcomes
def list_outcomes(system_size):
    # Define the set of characters to choose from
    characters = ['+', '-']
    
    # Use itertools.product to generate all possible combinations
    all_combinations = itertools.product(characters, repeat=system_size)
    
    # Convert tuples to strings
    all_strings = [''.join(combination) for combination in all_combinations]
    
    return all_strings
    

# Numerically simulating the prob of getting outcomes in a measurement scheme
def generate_prob(rho, measurement_scheme, measurement_outcome):
    if len(measurement_scheme)!=len(measurement_outcome):
        print("Lengths do not match!")
    else:
        proj_list=[(1/2)*(qt.identity(2)+dict_outcomes[measurement_outcome[i]]*dict_pauli_operators[measurement_scheme[i]]) for i in range(len(measurement_scheme))]
        povm=qt.tensor(proj_list)
        prob=(rho*povm).tr()
    return prob.real
        
def generate_prob_scheme(rho, measurement_scheme, system_size):
    # Initialize prob
    prob=np.zeros(2**len(measurement_scheme))
    all_outcomes=list_outcomes(system_size)
    
    for i in range(2**len(measurement_scheme)):
        prob[i]=generate_prob(rho, measurement_scheme, all_outcomes[i])
    return prob

# Generating the data set: ['XX++','xY+-',...]
def generate_data_list(rho, all_measurements, system_size):    
    data=[]
    all_outcomes=list_outcomes(system_size)
    
    for measurement_scheme in all_measurements:
        prob=generate_prob_scheme(rho, measurement_scheme, system_size)
        
        single_outcome=np.random.choice(all_outcomes, p=prob)
        data.append(''.join([measurement_scheme,single_outcome]))
        
    return data