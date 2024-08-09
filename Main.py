'''
--------------------------------------------------------------------------------
This code receives a (universally general) unitary Clifford+T input circuit
(supplied as a .qasm file) and performes one of two tasks:

- Task 1 (PBC compilation): The code compiles the input quantum circuit using
Pauli-based computation (PBC) [1,2] at the same time that it makes a weak simu-
lation of the original circuit (via the compiled circuit).
Input  : Clifford+T quantum circuit with n qubits, t T gates and m measurements
Output : The outcome of the simulation, performed via an interleaved procedure
where compilation and computation are done interchangeably. The compilation is
hidden within the whole procedure, and comes from the fact that at most t
independent and pairwise commuting t-qubit Pauli operators need to be measured
to carry out the whole computation [1].

- Task 2 (Hybrid PBC): The code estimates the probability that the output bit
yields the outcome s=1 by simulating a number k of virtual qubits. To estimate
this probability within a maximum relative error of ɛ, we need to run the exper-
iment a certain number of shots, N(ɛ, k), which is suitably determined by our
code.
Input  : Clifford+T quantum circuit with n qubits, t T gates and 1 measurement
Output : A list with the N values obtained for the random variable ξ whose ex-
pectation value E(ξ) is an unbiased estimator for the desired probability [1].
Thus, from these results, we get an estimate of the desired probability, p, and
can also define suitable confidence intervals.

--------------------------------------------------------------------------------
It is also important to note that, in order to perform the computation, we need
to translate the multi-qubit Pauli operator to be measured in each computational
layer back into a quantum circuit containing only single-qubit Z-measurements.
We do this using a new proposal different from that presented in [2], with the
advantage of being more practical and having a better asymptotic scaling of the
total number of gates [3].

Useful references:
[1]  Sergey Bravyi, Graeme Smith, and John A. Smolin. “Trading Classical and
Quantum Computational Resources”. In:Phys. Rev.X 6 (2016), p. 021043.
DOI: https://doi.org/10.1103/PhysRevX.6.021043.
EPRINT: https://arxiv.org/abs/1506.01396.

[2]  Mithuna Yoganathan, Richard Jozsa, and Sergii Strelchuk. “Quantum advantage
of unitary Clifford circuits with magic state inputs”. In:Proc. R. Soc. A475
(2019),p. 20180427. DOI: https://www.doi.org/10.1098/rspa.2018.0427.

[3]  Filipa C. R. Peres and Ernesto F. Galvao. "Quantum circuit compilation and
hybrid computation using Pauli-based computation". In:Quantum 7, 1126 (2023).
DOI: https://doi.org/10.22331/q-2023-10-03-1126.

[3] Filipa C. R. Peres and Ernesto F. Galvao. "Reducing Depth and Measurement
Weights in Pauli-based Computation".
EPRINT: https://arxiv.org/abs/2408.04007.

Author: F.C.R. Peres
Creation date: 01/06/2021
Last update: 09/08/2024
--------------------------------------------------------------------------------
'''
import c_and_c as cc

# Select the location and name of the input sample file(s)
file_loc = '/Users/folder/'
input_file_name = 'circuit_file'

# Parameter `go` for the greedy algorithm
go = 1

# Select the number of virtual qubits (place `0` to carry out PBC compilation)
vq = 0
prec = 0.01
cl = 0.99

# The user can change the following names at will
clifford_file_name = 'MSI'
output_file_name = 'Compilation_data'
resources_file_name = 'Resources_data'

if vq == 0:
    # Run PBC compilation (Task 1 in [3]):
    cc.run_pbc(file_loc,
               input_file_name,
               clifford_file_name,
               output_file_name,
               resources_file_name,
               shots=1024,
               paths_to_file=32,
               dummy=False,
               greedy_order=go,
               plot_hist=True,
               norm_hist=True)

else:
    # Run hybrid PBC (Task 2 in [3])
    cc.hybrid_pbc(file_loc,
                  input_file_name,
                  clifford_file_name,
                  resources_file_name,
                  greedy_order=go,
                  virtual_qubits=vq,
                  precision=prec,
                  confidence_level=cl)
