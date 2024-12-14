# Krylov shadow tomography: Efficient estimation of quantum Fisher information

This repository is a companion to the research paper with the same title (which will be posted on arXiv soon).

# Structure of this repository

- The two files `data_acquisition` and `prediction_QFI` are used to numerically simulate the Krylov shadow tomography (KST) proposed in the accompanying paper, where we show how to predict $\hat{B}_1^{(\mathsf{Kry})}$.
- The file `my_functions` is used to define various functions needed in the numerical simulation of the KST.
- The file `app1_RE_hat_vs_p` is associated with the application 1 in the accompanying paper. Here we study the relative error $\hat{\mathcal{E}}$ as a function of $p$, which is estimated by numerically simulating the KST.
- The file `app1_M_vs_N` is associated with the application 1 in the accompanying paper. Here we study the required number of classical shadows, $M$, as a function of $N$ (which is the number of qubits).
- The file `app1_fig2c_linear_fit` is associated with the application 1 in the accompanying paper. Here, based on some data obtained, we use linear fit to predict the scaling of $M$ in relation to $N$.
- The file `figure_comparison_app1` is associated with the application 1 in the accompanying paper. This is used to draw Figure 2 in the accompanying paper.
- The file `app2_RE_hat_vs_k` is associated with the application 2 in the accompanying paper. Here we study the relative error $\hat{\mathcal{E}}$ as a function of $k$, which is estimated by numerically simulating the KST.
- The file `app2_M_vs_N` is associated with the application 2 in the accompanying paper. Here we study the required number of classical shadows, $M$, as a function of $N$ (which is the number of qubits).
- The file `app2_fig3c_linear_fit` is associated with the application 2 in the accompanying paper. Here, based on some data obtained, we use linear fit to predict the scaling of $M$ in relation to $N$.
- The file `figure_comparison_app2` is associated with the application 2 in the accompanying paper. This is used to draw Figure 3 in the accompanying paper.
- All the data we generated and used in the accompanying paper can be found in the folder `data`.

# Basic requirement 

Apart from some commonly used packages, e.g., `numpy`, `json`, `itertools`, `math`, and `matplotlib`, the following two packages are required.

- The package `qutip` should be installed. See [this website](https://qutip.readthedocs.io/en/qutip-5.0.x/installation.html) for guidance.

- The package `scikit-learn` should be installed. See [this website](https://scikit-learn.org/1.5/install.html) for guidance.



