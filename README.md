# Circuit compilation and hybrid computation using Pauli-based computation

In this repository we can find the Python code used to produce the results
presented in our pre-print [https://arxiv.org/abs/2203.01789].

This code uses different modules, each equipped with different classes, methods
and functions to make things more readable and user friendly.
Each module is suitably documented at the start for clarity. In principle,
users need only open and make changes to `Main.py`.

Contact: filipa.peres@inl.int
-------------------------------------------------------------------------------


**Prerequisites:** This code was written using the following versions:
* Python 3.7.10;
* numpy 1.20.1;
* matplotlib 3.3.4;
* 'qiskit-terra': '0.17.4', 'qiskit-aer': '0.8.2', 'qiskit-ignis': '0.6.0',
'qiskit-ibmq-provider': '0.13.1', 'qiskit-aqua': '0.9.1', 'qiskit': '0.26.2',
'qiskit-nature': None, 'qiskit-finance': None, 'qiskit-optimization': None,
'qiskit-machine-learning': None


**Brief description:** The modules supplied allow us to execute two different
tasks (as described in our pre-print []):
* Task 1: Efficient circuit compilation and weak simulation; to carry out this
task one should open the `Main.py` module, change the number of virtual qubits,
`vq`, to 0 and adjust the parameters of the function `cc.run_pbc` as desired;
* Task 2: Hybrid computation using virtual qubits and approximate strong simu-
lation with maximum relative error ϵ; to perform this task one should open the
`Main.py` module, change the number of virtual qubits, `vq`, to a number greater
than 0, and adjust the parameters of the function `cc.hybrid_pbc` as desired.

 
**Use instructions**:
1. Make sure that the four Python modules (`Main.py`, `input_prep_t1.py`,
`input_prep_t2.py`, and `c_and_c.py`) are all in the same directory in your
computer;
2. Copy the input (.qasm) toy files supplied in this repository into a folder
in your computer (or make your own files with suitable Clifford+T quantum
circuits);
3. The files output by the code will be saved to a folder named "output" which
will be found inside the folder where you have placed the input file or files;
4. In the `Main.py` file change the location of the input files to point to the
correct location in your computer; change also the name of the input file(s)
appropriately, and choose the desired name for the different output files.
Adjust the parameters of `cc.run_pbc` or `cc.hybrid_pbc` depending on the
simulation you want to run;
5. Open a terminal window in the directory where you have placed the Python
modules and run the command: `python Main.py`;
6. Check the output files at the location that you have selected.


**Copyright:**
Copyright (C) 2022  F.C.R. Peres

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.