"""
Using Linear Regression to predict the scaling of M w. r. t. N.

Application1
"""

import numpy as np
import json
from sklearn.linear_model import LinearRegression


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
Linear Regression
"""
M_list_log=np.array(np.log2(M_list))

N_list=N_list.reshape(5, 1)
M_list=M_list_log.reshape(5, 1)

reg = LinearRegression().fit(N_list, M_list_log)


# M= a*2**(b*N)

print("a is", reg.intercept_)
print("b is", reg.coef_)
