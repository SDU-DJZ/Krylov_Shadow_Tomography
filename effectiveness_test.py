import os
import json  # Added the missing import for json
import numpy as np
import qutip as qt
import my_functions as mf
from multiprocessing import Pool

# Set the current working directory
os.chdir('C:\\Users\\dell\\Dropbox\\KST\\Codes')

"""
Warning: update the round label before running the program
"""
round_label = 8

# Global parameters
N = 2
Test_number = 10000000
H = mf.generate_hamiltonian(N)

# Function to process each test
def process_test(i):
    # Generate random density matrix
    rho = mf.generate_random_dm(N)
    
    # Calculate quantum Fisher information and lower bounds
    QFI = mf.quantum_fisher_information(rho, H)
    LB1 = mf.lower_bound(rho, H, 1, N)
    LB2 = mf.lower_bound(rho, H, 2, N)
    LB3 = mf.lower_bound(rho, H, 3, N)
    LB_Leg = mf.lower_bound_leg(N, rho)
    LB_Sub = mf.lower_bound_sub(rho, H)
    K1 = mf.Krylov_QFI(rho, H, 1, N)
    K2 = mf.Krylov_QFI(rho, H, 2, N)
    K3 = mf.Krylov_QFI(rho, H, 3, N)

    # Initialize result counters for this test
    results = {
        'QFI': QFI > N,
        'LB1': LB1 > N,
        'LB2': LB2 > N,
        'LB3': LB3 > N,
        'LB_Leg': LB_Leg > N,
        'LB_Sub': LB_Sub > N,
        'K1': K1 > N,
        'K2': K2 > N,
        'K3': K3 > N
    }
    
    return results

# Function to aggregate results
def aggregate_results(results_list):
    # Initialize counters
    N_Total = N_LB1 = N_LB2 = N_LB3 = N_LB_Leg = N_LB_Sub = N_K1 = N_K2 = N_K3 = 0
    
    # Aggregate the results
    for result in results_list:
        N_Total += result['QFI']
        N_LB1 += result['LB1']
        N_LB2 += result['LB2']
        N_LB3 += result['LB3']
        N_LB_Leg += result['LB_Leg']
        N_LB_Sub += result['LB_Sub']
        N_K1 += result['K1']
        N_K2 += result['K2']
        N_K3 += result['K3']

    return {
        "N_Total": N_Total,
        "N_LB1": N_LB1,
        "N_LB2": N_LB2,
        "N_LB3": N_LB3,
        "N_LB_Leg": N_LB_Leg,
        "N_LB_Sub": N_LB_Sub,
        "N_K1": N_K1,
        "N_K2": N_K2,
        "N_K3": N_K3
    }

# Function to convert NumPy types to Python types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert NumPy integers to Python ints
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert NumPy floats to Python floats
    return obj

# Create a pool of processes for parallel computation
if __name__ == '__main__':
    with Pool(processes=28) as pool:  # Use 28 processes to match the number of cores
        # Distribute the tasks across processes
        results = pool.map(process_test, range(Test_number))

    # Aggregate results from all processes
    aggregated_results = aggregate_results(results)
    
    new_results = {f'data{round_label}': {'system size': N, 'aggregated results': aggregated_results}}

    # Check if the file exists and load old results
    if os.path.exists('numerical_results_sm_effectiveness_test.json'):
        with open('numerical_results_sm_effectiveness_test.json', 'r', encoding='utf-8') as file:
            old_results = json.load(file)
    else:
        old_results = {}
    
    # Convert NumPy types in the results to standard Python types
    combined_results = {**old_results, **convert_numpy_types(new_results)}

    # Save the combined results to the file
    with open('numerical_results_sm_effectiveness_test.json', 'w', encoding='utf-8') as file:
        json.dump(combined_results, file, ensure_ascii=False, indent=2)
        
    # Print the results
    print("The number of tests passed through QFI-based criterion:", aggregated_results["N_Total"])
    print("The number of tests passed through LB1-based criterion:", aggregated_results["N_LB1"])
    print("The number of tests passed through LB2-based criterion:", aggregated_results["N_LB2"])
    print("The number of tests passed through LB3-based criterion:", aggregated_results["N_LB3"])
    print("The number of tests passed through K1-based criterion:", aggregated_results["N_K1"])
    print("The number of tests passed through K2-based criterion:", aggregated_results["N_K2"])
    print("The number of tests passed through K3-based criterion:", aggregated_results["N_K3"])
    print("The number of tests passed through Leg-based criterion:", aggregated_results["N_LB_Leg"])
    print("The number of tests passed through Sub-based criterion:", aggregated_results["N_LB_Sub"])

    print("The efficiency of LB1-based criterion:", aggregated_results["N_LB1"] / aggregated_results["N_Total"])
    print("The efficiency of LB2-based criterion:", aggregated_results["N_LB2"] / aggregated_results["N_Total"])
    print("The efficiency of LB3-based criterion:", aggregated_results["N_LB3"] / aggregated_results["N_Total"])
    print("The efficiency of Leg-based criterion:", aggregated_results["N_LB_Leg"] / aggregated_results["N_Total"])
    print("The efficiency of Sub-based criterion:", aggregated_results["N_LB_Sub"] / aggregated_results["N_Total"])
    print("The efficiency of K1-based criterion:", aggregated_results["N_K1"] / aggregated_results["N_Total"])
    print("The efficiency of K2-based criterion:", aggregated_results["N_K2"] / aggregated_results["N_Total"])
    print("The efficiency of K3-based criterion:", aggregated_results["N_K3"] / aggregated_results["N_Total"])
