import os
import numpy as np
import json
import multiprocessing
import my_functions as mf
import data_acquisition as da
import prediction_QFI as pr

os.chdir(r'C:\Users\dell\Dropbox\2024-KSE\Codes')

"""
Warning: update the round label before running the program
"""
round_label = 11

# Number of qubits
N = 6
# Number of points in Plots for a given N
points_num = 3
# Number of repetition
repeat_num = 1
# Generate p
k_list=list(range(1,points_num+1))


# Hamiltonian 
H = mf.generate_hamiltonian(N)

# Number of batches
B1 =16
B2 =40
L = 5000
# Total number of measurements: M=BL
M = B1*B2*L


def calculate_errors(i):
    error_list = np.zeros(points_num)
    # Randomly select M Pauli measurements
    all_measurements = da.randomized_measurement_procedure(M, N)
    for j in range(points_num):        
        rho = mf.generate_bound_entangled_state(N, k_list[j])
        # Exact value of QFI
        QFI = mf.quantum_fisher_information(rho, H)
    
        """
        Data acquisition phase
        """            
        # Acquire measurement data
        measurement_data = da.generate_data_list(rho, all_measurements, N)
        
        """
        Prediction phase
        """
        # Split measurement data into B batches
        bs = pr.batch_shadows(measurement_data, B1*B2, L)
        # Predict the 1st-order krylov QFI
        F1_hat = pr.F1_hat(bs, B1, B2, H.full())
        error = np.abs(F1_hat - QFI) /(QFI)
        error_list[j]=error
    return error_list


if __name__ == "__main__":
    with multiprocessing.Pool(processes=28) as pool:
        results = pool.map(calculate_errors, range(repeat_num))
        
    # Initialize error_avg
    error_avg = np.zeros(points_num)
    for error_list in results:
        error_avg += error_list / repeat_num

    """
    Store numerical results
    """
    error_avg_serializable = error_avg.tolist()
    

    new_results = {f'data{round_label}': {'system_size': N, 'average_errors': error_avg_serializable, 'B1, B2, and L are': [B1, B2, L]}}

    if os.path.exists('numerical_results_app2_k.json'):
        with open('numerical_results_app2_k.json', 'r', encoding='utf-8') as file:
            old_results = json.load(file)
    else:
        old_results = {}

    combined_results = old_results | new_results

    with open('numerical_results_app2_k.json', 'w', encoding='utf-8') as file:
        json.dump(combined_results, file, ensure_ascii=False, indent=2) 
    

