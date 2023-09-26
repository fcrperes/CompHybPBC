'''
--------------------------------------------------------------------------------
"c_and_c" stands for "compilation and computation"

This module contains all the tools necessary to carry out the compilation of
Clifford+T quantum circuits using Pauli-based computation (PBC) as well as to
perform hybrid computation.

The circuit to be input for this module is the adaptive msi Clifford circuit
prepared using either the module `input_prep_t1.py` or `input_prep_t2.py` Python
modules, depending on whether we are to carry out Task 1 (i.e., PBC compilation)
or Task 2 (hybrid PBC).
Note that there are 2 quantum registers: one named `q_stab` and the other named
`q_magic`. The latter is the register that contains the qubits which are in the
magic state |A⟩, and which are the only ones that will be used in the actual
quantum part of the computation.

* Pauli_Operator class: contains a series of methods which can be applied to any
Pauli_Operator instance. The methods are pretty straightforward and well
documented below so we abstain from further repetition in here. It is, however,
important to note that Pauli_Operator instances refer to Hermitian Pauli
operators which are characterized by a binary vector with 2n+1 bits, where the
last bit registers the signal of that Pauli (either + (if the bit is 0) or - (if
the bit is 1)) [1].

* qu_computing(): is the function that carries out the actual compiled quantum
computation associated with the Clifford+T quantum circuit that we started with.
At this point, since "dynamic circuits" are not yet available at IBM's Quantum
Experience, the function uses Qiskit's Statevector Simulator. Once that
functionality is unlocked we need only re-write this function of the code to
run the compiled computation on an actual quantum processor. For now, this
serves as a proof of concept. If the "dummy" option is set to True, we replace
the actual quantum simulator by a naïve (dummy) simulator that just coin-tosses
the outcome of each measurement. In this case, the quantum computation is
neither being carried out nor simulated. This is used simply to test the time of
the classical processing in our code for larger quantum circuits, and also to
provide some idea of the numerics for the depth and gate counts of the compiled
computation for those larger instances. However, the output distribution
obtained when this option is used will be incorrect whenever the quantum
measurements do not have uniform output: p(0)=p(1)=1/2.

* run_pbc(): carries out Task 1 [2].

* hybrid_pbc(): carries out Task 2 [2].

Useful references:
[1]  S.Aaronson  and  D.Gottesman. “Improved simulation of stabilizer circuits”.
In:Phys. Rev. A70 (2004), p. 052328.
DOI: https://doi.org/10.1103/PhysRevA.70.052328.
EPRINT: https://arxiv.org/abs/quant-ph/0406196v5.

[2]  Filipa Peres and Ernesto Galvão. “Quantum circuit compilation and hybrid
computation using Pauli-based computation”.
EPRINT: https://arxiv.org/abs/2203.01789.


Author: F.C.R. Peres
Creation date: 14/06/2021
Last updated: 26/09/2023
--------------------------------------------------------------------------------
'''

import os
import json
import copy
import random
import time
import math
from math import sqrt
import statistics as stat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import qiskit
import qiskit.providers.aer as aer

import input_prep_t1 as inp1
import input_prep_t2 as inp2


class Pauli_Operator():
    """This class represents Hermitian Pauli operators and defines several
    operations that can be performed on and with them.
    """
    def __init__(self, bin_vec):
        """Initialize the attributes that are used to describe each
        Pauli_Operator instance (i.e., each Hermitian Pauli operator).

        Args:
            bin_vec (list([0/1])): the binary Python list that characterizes
            the Pauli_Operator instance
        """
        # The attributes of any Pauli_Operator instance are, the corresponding
        # binary list and the number of elements in that list
        self.bin_vec = bin_vec
        self.nr_columns = len(bin_vec)

    def h(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated through a Hadamard gate acting on the i-th
        qubit of the `q_stab` quantum register. [1]

        Args:
            i (int): number of the qubit the Hadamard is being applied to
        """
        self.bin_vec[-1] = (self.bin_vec[-1] +
                            self.bin_vec[i] * self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)]) % 2
        if (self.bin_vec[i] !=
                self.bin_vec[i + int((self.nr_columns - 1) / 2)]):
            aux = self.bin_vec[i]
            self.bin_vec[i] = self.bin_vec[i + int((self.nr_columns - 1) / 2)]
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = aux

    def s(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated to the RIGHT of a phase gate S acting on the
        i-th qubit of the `q_stab` quantum register. [1]
        (Observation: left and right here are with respect to the circuit model
        depiction)

        Args:
            i (int): number of the qubit the S gate is being applied to
        """
        self.bin_vec[-1] = (self.bin_vec[-1] +
                            self.bin_vec[i] * self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)]) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def sdg(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated to the LEFT of a phase gate S acting on the
        i-th qubit of the `q_stab` quantum register. This is the transformation
        to use (instead of `s(i)`) when propagating the Pauli measurements from
        the end of the circuit to the beginning. In the Hadamard and controlled-
        NOT cases there is no distinction because those two are Hermitian
        Clifford operators, but in the case of the phase gate we need to be very
        careful with what we are doing.

        Args:
            i (int): nr. of the qubit the S^{dagger} gate is being applied to
        """
        self.bin_vec[-1] = (self.bin_vec[-1] + self.bin_vec[i] *
                            (self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)] + 1)) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def cx(self, i, j):
        """This method updates the Pauli_Operator instance it is applied to when
        that instance is propagated through a controlled-NOT gate CX controlled
        on the i-th qubit of the `q_stab` quantum register and targetting the
        j-th qubit of the system. [1]

        Args:
            i (int): number of the control qubit
            j (int): number of the target qubit
        """
        self.bin_vec[-1] = (self.bin_vec[-1] + self.bin_vec[i] *
                            self.bin_vec[j + int((self.nr_columns - 1) / 2)] *
                            (self.bin_vec[j] + self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)] + 1)) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[j] = (self.bin_vec[j] + 1) % 2
        if self.bin_vec[j + int((self.nr_columns - 1) / 2)] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def v(self, P, Q, sP, sQ):
        """This method updates the Pauli_Operator instance it is applied to when
        propagating it to the left of the Clifford V = (sP*P + sQ*Q)/sqrt(2).

        Args:
            P (Pauli_Operator): Pauli operator instance P
            Q (Pauli_Operator): Pauli operator instance Q
            sP (int): Measurement outcome obtained for the Pauli operator P
            sQ (int): Measurement outcome obtained for the Pauli operator Q
        """
        symprodP = self.symplectic_inner_product(P)
        symprodQ = self.symplectic_inner_product(Q)

        if (symprodP == 0 and symprodQ == 1):
            RPQ = self.RPQproduct(P, Q)
            self.bin_vec = RPQ.bin_vec

            self.bin_vec[-1] = (self.bin_vec[-1] + sP + sQ) % 2

        elif (symprodP == 1 and symprodQ == 0):
            RPQ = self.RPQproduct(P, Q)
            self.bin_vec = RPQ.bin_vec

            self.bin_vec[-1] = (self.bin_vec[-1] + sP + sQ + 1) % 2

        elif (symprodP == 1 and symprodQ == 1):
            # P and Q both anti-commute with the Pauli_Operator instance
            self.bin_vec[-1] = (self.bin_vec[-1] + 1) % 2

    def gen_update(self, gate_sequence, q_count):
        """This method automatically updates the Pauli_Operator instance under
        consideration by propagating it through the sequence of Clifford
        unitaries specified in the `gate_sequence` list.

        Args:
            gate_sequence (list[str]): The list that establishes the gate
            sequence through which the Pauli needs to be propagated; this list
            has elements that are strings 'h (...)', 's (...)', 'cx (...)', or
            'v (...)'; each line has a certain default format and the knowledge
            of that format can be used to read the information encoded on it.
            q_count (int): The number of qubits in the original Clifford+T
            quantum circuit
        """
        for line in gate_sequence:
            if line.startswith('h '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.h(i)

            elif line.startswith('s '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.sdg(i)

            elif line.startswith('cx '):
                w = line.partition('q_stab[')
                x = w[2].partition('],')
                i = int(x[0])
                y = x[2].partition('[')
                z = y[2].partition(']')
                if y[0] == 'q_stab': j = int(z[0])
                elif y[0] == 'q_magic': j = int(z[0]) + q_count
                self.cx(i, j)

            elif line.startswith('v '):
                v = line.partition(' ')
                w = v[2].partition(' ')
                sP = int(w[0])
                x = w[2].partition(' ')
                sQ = int(x[0])
                y = x[2].partition(' ')
                P = Pauli_Operator([int(y[0][i]) for i in range(len(y[0]))])
                z = y[2].partition(';')
                Q = Pauli_Operator([int(z[0][i]) for i in range(len(z[0]))])
                self.v(P, Q, sP, sQ)

    def return_Pauli_vec(self):
        """Returns the binary vector associated with the Pauli_Operator instance

        Returns:
            list([0/1]): the binary vector associated with the updated Pauli
        """
        return self.bin_vec

    def return_Pauli_str(self):
        """Returns the list of single-qubit Pauli operators that correspond to
        the Pauli_Operator instance.
        (Mostly useful for functionality testing, debugging and output reading.)

        Returns:
            str: String indicating the sequence of single-qubit Pauli operators
            (I, X, Y, Z) that constitute the Pauli_Operator instance
        """
        Pauli_str = ''
        if self.bin_vec[-1] == 0: Pauli_str += '+'
        else: Pauli_str += '-'

        for i in range(int((self.nr_columns - 1) / 2)):
            if (self.bin_vec[i] == 0
                    and self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 0):
                Pauli_str += 'I'
            elif (self.bin_vec[i] == 0
                  and self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 1):
                Pauli_str += 'Z'
            elif (self.bin_vec[i] == 1
                  and self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 0):
                Pauli_str += 'X'
            else:
                Pauli_str += 'Y'

        return Pauli_str

    def symplectic_inner_product(self, P):
        """Method that calculates the symplectic inner product between the
        Pauli_Operator instance to which this method is applied, and the
        Pauli_Operator instance P.

        Args:
            P (Pauli_Operator): The Pauli operator instance with which we want
            to perform the symplectic inner product

        Returns:
            int (0/1): The value of the symplectic inner product between the
            Pauli operator instance this method is applied to and the input
            Pauli operator instance P
        """
        P1 = self.bin_vec.copy()
        P2 = P.bin_vec.copy()
        P1.pop(-1)
        P2.pop(-1)

        symprod = 0
        for i in range(int((self.nr_columns - 1) / 2)):
            symprod = (symprod +
                       (P1[i] * P2[int((self.nr_columns - 1) / 2) + i] +
                        P1[int((self.nr_columns - 1) / 2) + i] * P2[i])) % 2

        return symprod

    def RPQproduct(self, P, Q):
        """This method calculates the triple product between the Pauli_Operator
        instance R to which it is applied (i.e., `self`) and the Pauli instances
        P and Q, that is, it calculates the hermitian Pauli_Operator instance
        R' = PQR. Note that R' will be Hermitian because this PQR product will
        only be applied in situations wherein P and Q anti-commute, and R anti-
        commutes with either P or Q (but not both simultaneously). In situations
        other than this specific one this PQR product is not valid and should
        not be employed.

        Args:
            P (Pauli_Operator): Pauli operator instance P
            Q (Pauli_Operator): Pauli operator instance Q

        Returns:
            Pauli_Operator: the Hermitian Pauli_Operator instance R'=PQR.
        """
        PQ_prod = [0] * self.nr_columns
        RPQ_prod = [0] * self.nr_columns

        # start by multiplying P and Q:
        for i in range(self.nr_columns - 1):
            PQ_prod[i] = (P.bin_vec[i] + Q.bin_vec[i]) % 2

        contribution = 0
        for i in range(int((self.nr_columns - 1) / 2)):
            if (P.bin_vec[i] == 1
                    and P.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                if (Q.bin_vec[i] == 1 and
                        Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1
                elif (Q.bin_vec[i] == 0
                      and Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

            elif (P.bin_vec[i] == 1
                  and P.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (Q.bin_vec[i] == 1 and
                        Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 3
                elif (Q.bin_vec[i] == 0
                      and Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1

            elif (P.bin_vec[i] == 0
                  and P.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (Q.bin_vec[i] == 1 and
                        Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 1
                elif (Q.bin_vec[i] == 0
                      and Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

        PQ_prod[-1] = (2 * P.bin_vec[-1] + 2 * Q.bin_vec[-1] +
                       contribution) % 4

        # next calculate R*(PQ):
        for i in range(self.nr_columns - 1):
            RPQ_prod[i] = (self.bin_vec[i] + PQ_prod[i]) % 2

        contribution = 0
        for i in range(int((self.nr_columns - 1) / 2)):
            if (self.bin_vec[i] == 1
                    and self.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                if (PQ_prod[i] == 1
                        and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1
                elif (PQ_prod[i] == 0
                      and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

            elif (self.bin_vec[i] == 1
                  and self.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (PQ_prod[i] == 1
                        and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 3
                elif (PQ_prod[i] == 0
                      and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1

            elif (self.bin_vec[i] == 0
                  and self.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (PQ_prod[i] == 1
                        and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 1
                elif (PQ_prod[i] == 1
                      and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

        RPQ_prod[-1] = (2 * self.bin_vec[-1] + PQ_prod[-1] + contribution) % 4

        if RPQ_prod[-1] == 2: RPQ_prod[-1] = 1

        return Pauli_Operator(RPQ_prod)

    def is_independent(self, q_count, t_count, pipc_Paulis, M):
        """This method checks whether or not the Pauli operator it is applied to
        is independent from all previous (independent) Paulis in the Pauli-based
        computation.

        Args:
            q_count (int): The number of qubits in the original Clifford+T
            quantum circuit
            t_count (int): The number of T gates in the original Clifford+T
            quantum circuit
            pipc_Paulis (list([Pauli_Operator])): A Python list with all of the
            previously measured (independent and pairwise commuting)
            Pauli_Operator instances
            M: The matrix we use to compute the nullspace(), and which is
            changed and updated as the PBC procedure progresses

        Returns:
            tuple (diverse, list): (1) Admits 3 possible situations:
            (1a) int (0/1): the outcome of the Pauli if it is trivial
            (deterministic)
            (1b) 'True': if the Pauli is independent from all previous Paulis
            (1c) list([0/1]): a list of zeros and ones which is the kernel of
            the matrix made up by the previous (independent) Paulis and the
            current one. This list identifies which are the previous Pauli
            operators that the current Pauli depends on.
            (2) The matrix M is also returned by this function as it needs to
            be tracked during the whole computation
        """
        if (self.bin_vec[q_count:q_count + t_count]
                == [0 for _ in range(t_count)]
                and self.bin_vec[2 * q_count + t_count:-1]
                == [0 for _ in range(t_count)]):
            # if the Pauli is trivial in the ancillary (magic) qubits, its mmt
            # is (immediately) trivial and deterministic
            outcome = self.bin_vec[-1]
            return (int(outcome), M)

        else:
            # if the Pauli is not trivial in the ancillas it may be dependent
            # or it may be independent
            if len(pipc_Paulis) == q_count:
                #if this is the case we get the first independent Pauli:
                P = self.bin_vec.copy()
                M.append(P[q_count:q_count + t_count] +
                         P[2 * q_count + t_count:] + [1] + [0] * t_count)
                return ('True', M)

            else:
                P = self.bin_vec.copy()
                position = len(M)
                vec = [0] * (t_count + 1)
                vec[position] = 1
                M.append(P[q_count:q_count + t_count] +
                         P[2 * q_count + t_count:] + vec)

                (kernel, M) = nullspace(M)

                if kernel == []:
                    # The operator is independent from the previous operators.
                    return ('True', M)
                else:
                    return (kernel, M)

    def is_commuting(self, q_count, t_count, pipc_Paulis):
        """This method checks out whether the Pauli_Operator instance commutes
        with all previously measured (independent and pairwise commuting) Paulis
        in the Pauli-based computation

        Args:
            q_count (int): The number of qubits in the original Clifford+T
            quantum circuit
            t_count (int): The number of T gates in the original Clifford+T
            quantum circuit
            pipc_Paulis (list([Pauli_Operator])): A Python list with all of the
            previously measured (independent and pairwise commuting)
            Pauli_Operator instances

        Returns:
            'True': if the Pauli operator commutes with all previous Paulis
            int: the index of the first previous Pauli operator with which the
            current Pauli anti-commutes
        """
        for index in range(0, q_count):
            if self.bin_vec[index] == 1:
                # The Pauli operator anti-commutes at least with Z_{index}
                return index

        P = self.bin_vec.copy()
        P = P[q_count:q_count + t_count] + P[2 * q_count + t_count:]

        reduced_pipc_Paulis = []
        for Pauli in pipc_Paulis[q_count:]:
            Q = Pauli.bin_vec.copy()
            reduced_pipc_Paulis.append(
                Pauli_Operator(Q[q_count:q_count + t_count] +
                               Q[2 * q_count + t_count:]))

        count = q_count
        index = q_count
        for Pauli in reduced_pipc_Paulis:
            if Pauli_Operator(P).symplectic_inner_product(Pauli) == 1:
                break
            elif Pauli_Operator(P).symplectic_inner_product(Pauli) == 0:
                count += 1
            index += 1

        if count == len(pipc_Paulis):
            # The Pauli operator instance commutes with all previous Paulis
            return 'True'
        else:
            # The Pauli operator anti-commutes at least with the Pauli #{index}
            return index


def nullspace(matrix):
    """This function receives a matrix input as a list of list (i.e., a nested
    list) where each line represents a different binary vector and returns a
    list indicating which of the previously independent and pairwise communting
    Paulis can be used to obtain the current Pauli under test. If all vectors
    are linearly independent (mod 2), then the function returns an empty list.

    Args:
        matrix (list): nested list representing the matrix containing one
        binary vector per line

    Returns:
        tuple (list, list): (1) list with entries 0 or 1, or empty list. In the
        former case, the entries with value 1 indicate the Pauli operators
        which are linearly dependent on one another. (2) the updated matrix
        used to assess linear (in)dependence.
    """
    t_count = int((len(matrix[0]) - 2) / 3)
    n_rows = len(matrix)
    all_columns = [i for i in range(len(matrix[0]))]
    sel_columns = [i for i in range(2 * t_count)]
    unused_columns = copy.deepcopy(sel_columns)
    used_columns = []

    track_rc = [-1] * 2 * t_count  # tracks which column is 1 only for row r
    for r in range(n_rows - 1):
        for c in sel_columns:
            if matrix[r][c] == 1:
                track_rc[c] = r
                unused_columns.remove(c)
                used_columns.append(c)
                break

    for c1 in used_columns:
        if matrix[-1][c1] == 1:
            for c2 in all_columns:
                matrix[-1][c2] = (matrix[-1][c2] +
                                  matrix[track_rc[c1]][c2]) % 2

    if matrix[-1][:-(t_count + 2)] == [0] * (2 * t_count):
        # The current Pauli depends on previous Paulis:
        kernel = matrix[-1][2 * t_count + 1:2 * t_count + 1 + n_rows]
        matrix.pop()  # we eliminate the Pauli from the matrix
        return (kernel, matrix)

    else:
        # The current Pauli is independent from previous Paulis:
        kernel = []  # we leave the kernel empty to signal independence

        # The matrix will need to be modified so that this new Pauli is 1 in
        # a columns where all other Paulis are 0:
        for c1 in unused_columns:
            if matrix[-1][c1] == 1:
                track_rc[-1] = c1
                break

        flagged = track_rc[-1]
        for r in range(n_rows - 1):
            if matrix[r][flagged] == 1:
                for c in all_columns:
                    matrix[r][c] = (matrix[r][c] + matrix[-1][c]) % 2

        return (kernel, matrix)


def initialize_lists(q_count, t_count):
    """This function initializes the lists of Pauli_Operator instances and
    respective outcomes with which the PBC procedure is initiated. More
    specifically, it builds the Python lists with q_count-elements such that:
    list([Z_1,Z_2,...,Z_{q_count}]) and list([0,0,...,0])

    Args:
        q_count (int): The number of qubits in the original Clifford+T circuit
        t_count (int): The number of T gates in the original Clifford+T circuit

    Returns:
        tuple (list([Pauli_Operators]), list([0s])): The Python lists with which
        the PBC computation and compilation procedure is initiated.
    """
    Pauli_list = []
    outcome_list = []

    for row in range(q_count):
        bin_vec = [0] * (2 * (q_count + t_count) + 1)
        bin_vec[row + q_count + t_count] = 1
        Pauli_list.append(Pauli_Operator(bin_vec))
        outcome_list.append(0)

    return Pauli_list, outcome_list


def find_new_Pauli(circuit_list, q_count, t_count):
    """Function which reads the adaptive (gadgetized) Clifford circuit encoded
    in `circuit_list` and finds the next Pauli operator in the generalized PBC
    list sequence.

    Args:
        circuit_list (list[str]): List of operations associated with the
        adaptive Clifford circuit with magic state injection
        q_count (int): The number of qubits in the original Clifford+T circuit
        t_count (int): The number of T gates in the original Clifford+T circuit

    Returns:
        tuple (list[str], Pauli_Operator): Returns the updated circuit list, and
        the new Pauli_Operator instance.
    """
    bin_vec = [0] * (2 * (q_count + t_count) + 1)
    for line in circuit_list:
        if line.startswith('measure '):
            index = circuit_list.index(line)
            x = line.partition('[')
            y = x[2].partition(']')
            i = int(y[0])
            z = x[0].partition(' ')
            if z[2] == 'q_stab':
                bin_vec[i + q_count + t_count] = 1
            elif z[2] == 'q_magic':
                bin_vec[i + 2 * q_count + t_count] = 1
            circuit_list[index] = ''
            break

    Pauli = Pauli_Operator(bin_vec)
    gate_sequence = []
    i = 0
    for line in circuit_list:
        if i < index:
            i += 1
            if (line.startswith('h ') or line.startswith('s ')
                    or line.startswith('cx ') or line.startswith('v ')):
                gate_sequence.append(line)

    gate_sequence.reverse()
    Pauli.gen_update(gate_sequence, q_count)

    return circuit_list, Pauli


def qu_computing(q_count, t_count, current_Pauli, state_vector, dummy):
    """This is the function that interacts with the quantum simulator or carries
    out a naïve (dummy) 'simulation'.
    In fact, this is the only function that needs to be changed depending on the
    platform in which we are running the quantum circuit. In this case, the
    function is written to interact with Qiskit's Statevector simulator; but any
    user could change this function to suit their preferred platform.

    Args:
        q_count (int): The number of qubits in the original Clifford+T circuit
        t_count (int): The number of T gates in the original Clifford+T circuit
        current_Pauli (Pauli_Operator): The current Pauli_Operator instance to
        be measured
        state_vector (np.array([floats])): The state vector with which the
        quantum circuit must be initialized after the first operator has been
        measured.
        dummy (bool): Determines whether we use Qiskit's Statevector Simulator
        (if False) or if we use the naïve (dummy) 'simulation'.

    Returns:
        tuple (int, np.array([floats])): A tuple with (1) the outcome of the
        quantum measurement of the Pauli_Operator instance, and (2) the state of
        the system after the measurement
    """
    qr = qiskit.QuantumRegister(t_count, 'qr')
    cr = qiskit.ClassicalRegister(1, 'cr')

    if t_count > 1:
        qaux = qiskit.QuantumRegister(1, 'qaux')
        qc = qiskit.QuantumCircuit(qr, qaux, cr)

        if state_vector.size == 0:
            # For the first run, the initial state of the system is a tensor
            # product of magic states:
            for i in range(t_count):
                qc.h(i)
                qc.t(i)

            if (current_Pauli.bin_vec[q_count] == 1
                    and current_Pauli.bin_vec[q_count + int(
                        (current_Pauli.nr_columns - 1) / 2)] == 0):
                rm_layer = 1
            else:
                rm_layer = 2

        elif (state_vector.size != 0 and not dummy):
            # After the first run, we must initialize the circuit to the state
            # after the previous measurement:
            qc.initialize(state_vector, [i for i in range(t_count + 1)])
            qc.reset(qaux)
            rm_layer = 1  # we need to remove "initialize" from the depth value

        elif (state_vector.size != 0 and dummy):
            qc.reset(qaux)
            rm_layer = 0

        qc.h(qaux)

        for i in range(q_count, int((current_Pauli.nr_columns - 1) / 2)):
            j = i - q_count
            if (current_Pauli.bin_vec[i] == 0
                    and current_Pauli.bin_vec[i + int(
                        (current_Pauli.nr_columns - 1) / 2)] == 1):
                qc.h(qr[j])
                qc.cx(qaux, qr[j])
                qc.h(qr[j])

            elif (current_Pauli.bin_vec[i] == 1
                  and current_Pauli.bin_vec[i + int(
                      (current_Pauli.nr_columns - 1) / 2)] == 0):
                qc.cx(qaux, qr[j])

            elif (current_Pauli.bin_vec[i] == 1
                  and current_Pauli.bin_vec[i + int(
                      (current_Pauli.nr_columns - 1) / 2)] == 1):
                qc.s(qr[j])
                qc.s(qr[j])
                qc.s(qr[j])
                qc.cx(qaux, qr[j])
                qc.s(qr[j])

        qc.h(qaux)
        qc.measure(qaux, cr)
        depth = qc.depth() - rm_layer
        # the subtraction removes the appropriate value, for the correct
        # evaluation of the depth

    elif t_count == 1:
        # This is a simpler case where we only need to measure 1 qubit either in
        # the X, Y, or Z basis; there is no need for an ancilla or 2-qubit gates
        qc = qiskit.QuantumCircuit(qr, cr)

        qc.h(0)
        qc.t(0)

        if (current_Pauli.bin_vec[q_count] == 0
                and current_Pauli.bin_vec[-2] == 1):
            qc.measure(qr, cr)

        elif (current_Pauli.bin_vec[q_count] == 1
              and current_Pauli.bin_vec[-2] == 0):
            qc.h(0)
            qc.measure(qr, cr)

        elif (current_Pauli.bin_vec[q_count] == 1
              and current_Pauli.bin_vec[-2] == 1):
            qc.h(0)
            qc.s(0)
            qc.h(0)
            qc.measure(qr, cr)

        depth = qc.depth() - 2
        # the subtraction removes the gates H and T used to prepare |A⟩

    ops = qc.count_ops()

    if not dummy:
        # We run an actual quantum simulation if dummy is set to 'False':
        job = aer.StatevectorSimulator().run(qc)

        # Grab results from the job:
        result = job.result()
        state_vector = result.get_statevector()
        counts = result.get_counts()

        if '0' in counts.keys(): outcome = 0
        elif '1' in counts.keys(): outcome = 1

        if current_Pauli.bin_vec[-1] == 1: outcome = (outcome + 1) % 2
    else:
        # If dummy is set to 'True' we do a "dummy simulation" via a coin flip:
        outcome = random.randint(0, 1)
        # create a dummy state vector so that we trigger the correct conditional
        # state and have a correct depth calculation:
        state_vector = np.array([1])

    return outcome, state_vector, depth, ops


def compiling(circuit_list,
              q_count,
              t_count,
              first_gate_index,
              pipc_Paulis,
              matrix,
              pipc_outcomes,
              all_Paulis,
              all_outcomes,
              current_Pauli,
              first_qm,
              dummy=False):
    """This function carries out the PBC compilation procedure, calling
    'qu_computing()' when the outcome of the Pauli operator must be determined
    via an actual quantum measurement, and carring out the appropriate classical
    processing when that is not the case

    Args:
        circuit_list (list(str)): The circuit list
        q_count (int): The number of qubits in the original Clifford+T circuit
        t_count (int): The number of T gates in the original Clifford+T circuit
        first_gate_index (int): The index of the 1st gate on the `circuit_list`
        (this tells us where to insert the Clifford V(P,Q) in case that is
        necessary)
        pipc_Paulis (list(Pauli_Operator)): A list with all of the previously
        measured (independent and pairwise commuting) Pauli operators
        pipc_outcomes (list(0/1))): A list with the measurement outcomes
        associated with all of the previously measured Pauli operators
        matrix (): TBD
        all_Paulis (list(Pauli_Operator)): A list with all of the Pauli
        operators from the generalized PBC that have been processed up until
        this point
        all_outcomes (list(0/1))): A list recording the outcome associated with
        each of the Pauli operators from the generalized PBC that have been
        processed up to the current point
        current_Pauli (Pauli_Operator): The current Pauli operator instance in
        the generalized PBC which needs to be evaluated
        first_qm (bool): A boolean variable that registers whether the a first
        quantum measurement has been carried out or not, that is, whether a
        non-empty state vector exists or not.
        dummy (bool, optional): Determines whether we use Qiskit's Statevector
        Simulator (if False) or if we use the naïve (dummy) simulation. Defaults
        to False

    Returns:
        tuple (list[str], list([Pauli_Operators]), list([0/1]),
        list([Pauli_Operators]), list([0/1]), np.array([floats])): (1) The
        updated circuit list, (2)-(5) the lists which are input to the funtion
        incremented by the current Pauli (where appropriate) and (6) the
        state_vector (which is only changed with respect to the input if an
        actual quantum measurement was performed)
    """
    global quantum_depth, quantum_ops, job_time, tcr
    global tji, tt
    global state_vector

    job_time = 0
    quantum_depth = 0
    tcr = 0
    quantum_ops = {}

    Pauli_phases = [Pauli.bin_vec[-1] for Pauli in pipc_Paulis]

    index_ac_Pauli = current_Pauli.is_commuting(q_count, t_count, pipc_Paulis)
    if index_ac_Pauli != 'True':
        outcome = random.randint(0, 1)

        P = pipc_Paulis[index_ac_Pauli]
        sP = int(pipc_outcomes[index_ac_Pauli])
        Q = current_Pauli
        sQ = int(outcome)
        strP = ''
        strQ = ''
        for i in range(len(P.bin_vec)):
            strP += str(int(P.bin_vec[i]))
            strQ += str(int(Q.bin_vec[i]))
        # Clifford for circuit list:
        v = f'v {sP} {sQ} {strP} {strQ};\n'
        circuit_list.insert(first_gate_index, v)

    else:
        kernel, matrix = current_Pauli.is_independent(q_count, t_count,
                                                      pipc_Paulis, matrix)
        if isinstance(kernel, list):
            nr_rows = len(kernel)
            outcome = 0
            for i in range(nr_rows - 1):
                if kernel[i] == 1:
                    outcome = (outcome + pipc_outcomes[i + q_count] +
                               Pauli_phases[i + q_count]) % 2
            if current_Pauli.bin_vec[-1] == 1:
                outcome = (outcome + 1) % 2

        elif isinstance(kernel, int):
            outcome = kernel

        else:
            if not first_qm:
                state_vector = np.array([])
                tt = 0

                tji = time.perf_counter()
                outcome, state_vector, quantum_depth, quantum_ops = qu_computing(
                    q_count, t_count, current_Pauli, state_vector, dummy)
                tjf = time.perf_counter()

                job_time = tjf - tji  # time of the quantum job
                tcr = tji - tt  # time in-between quantum jobs
                tt = tjf

                first_qm = True

            elif first_qm:
                tji = time.perf_counter()
                outcome, state_vector, quantum_depth, quantum_ops = qu_computing(
                    q_count, t_count, current_Pauli, state_vector, dummy)
                tjf = time.perf_counter()

                job_time = tjf - tji  # time of the quantum job
                tcr = tji - tt  # time in-between quantum jobs
                tt = tjf

            pipc_Paulis.append(current_Pauli)
            pipc_outcomes.append(outcome)

    all_Paulis.append(current_Pauli)
    all_outcomes.append(outcome)

    for line in circuit_list:
        if line.startswith('if'):
            index = circuit_list.index(line)
            if outcome == 0:
                circuit_list[index] = ''
            elif outcome == 1:
                circuit_list[index] = line.partition(' ')[2]
            break

    return circuit_list, pipc_Paulis, matrix, pipc_outcomes, all_Paulis,\
        all_outcomes, first_qm


def run_pbc(file_loc,
            input_file_name,
            clifford_file_name,
            output_file_name,
            resources_file_name,
            shots=1024,
            paths_to_file=0,
            dummy=False,
            plot_hist=False,
            norm_hist=False):
    """Function that carries out the adaptive Pauli-based computation (PBC)
    corresponding to the input quantum circuit.

    Args:
        file_loc (str): String indicating the location of the input file
        input_file_name (str): String with the name of the input file
        clifford_file_name (str): String with the name of the generated
        adaptive, magic-state injected Clifford circuit
        output_file_name (str): String with the desired name for the ouput file
        resources_file_name (str): String with the desired name for the
        resource file
        shots (int, optional): Number of shots to carry out in the computation.
        Defaults to 1024.
        paths_to_file (int, optional): Number of paths of the PBC tree to be
        saved to a .txt file. Defaults to 0.
        dummy (bool, optional): A flag which determines if we are going to use
        a real Schrödinger-type simulator (False), or if we are using a dummy
        simulator which simply performs a coin-toss (True). Defaults to False.
        plot_hist (bool, optional): A flag determining whether or not the user
        wants to plot the histogram with the output distribution. Defaults to
        False.
        norm_hist (bool, optional): A flag determining whether or not the user
        wants to normalize the histogram so that the height of each bar
        corresponds to the probability of the respective outcome. Defaults to
        False.
    """
    # Creating the folder for the output files:
    if not os.path.exists(f'{file_loc}output'):
        os.mkdir(f'{file_loc}output')

    # STEP 1: Creates the gadgetized (i.e. adaptive) Clifford circuit which is
    # equivalent to the original universal quantum circuit.
    qc = inp1.QuCirc(file_loc, input_file_name)
    clifford, q_count, t_count, cx_count, hs_count, nr_mmts = qc.msi_circuit(
        file_loc, clifford_file_name)
    # q_count -> nr. of qubits of the original quantum circuit
    # t_count -> nr. of T-gates of the original quantum circuit
    # nr_mmts -> nr. of measurements of the (new) gadgetized circuit

    for line in clifford:
        if (line.startswith('h ') or line.startswith('s ')
                or line.startswith('cx ')):
            first_gate_index = clifford.index(line)
            break

    # STEP 2: Compilation and computation
    results = {}
    cx_count_cc, hs_count_cc = ([], [])
    qu_time, cl_time, qc_depth = ([], [], [])
    min_coherence, mean_coherence, max_coherence = ([], [], [])

    with open(f'{file_loc}output/{output_file_name}.txt', 'w') as file_object:
        file_object.write(f'Numpy version: {np.__version__}\n')
        file_object.write(f'Matplotlib version: {matplotlib.__version__}\n')
        file_object.write(f'Qiskit version: {qiskit.__qiskit_version__}\n\n')

        file_object.write(f'Total number of shots: {shots}\n')
        file_object.write(
            f'Number of shots recorded in this file: {paths_to_file}\n\n')

        for i in range(shots):
            initial_time = time.perf_counter()

            # the circuit list is copied:
            clist = clifford.copy()

            all_Ps, all_outcs = initialize_lists(q_count, t_count)
            pipc_Ps, pipc_outcs = initialize_lists(q_count, t_count)
            mat = []

            fqm = False
            quant_time = 0
            quant_depth = 0
            quant_ops = {}
            cohe_per_shot = []
            for j in range(nr_mmts):
                clist, current_Pauli = find_new_Pauli(clist, q_count, t_count)
                clist, pipc_Ps, mat, pipc_outcs, all_Ps, all_outcs, fqm = compiling(
                    clist,
                    q_count,
                    t_count,
                    first_gate_index,
                    pipc_Ps,
                    mat,
                    pipc_outcs,
                    all_Ps,
                    all_outcs,
                    current_Pauli,
                    fqm,
                    dummy=dummy)
                quant_time += job_time
                quant_depth += quantum_depth
                if (not quant_ops and quantum_ops):
                    quant_ops = quantum_ops
                elif (quant_ops and quantum_ops):
                    for key, value in quantum_ops.items():
                        if key in quant_ops.keys():
                            quant_ops[key] += value
                        else:
                            quant_ops[key] = value

                if tcr:
                    cohe_per_shot.append(tcr)

            cohe_per_shot.pop(0)

            # Building the dictionary with the computation's outcomes:
            computation_outcome = ''
            for j in range(len(all_outcs[q_count + t_count:])):
                computation_outcome += str(
                    int(all_outcs[q_count + t_count + j]))

            if not computation_outcome in results.keys():
                results.update({computation_outcome: 1})
            else:
                results[computation_outcome] += 1

            # once we have the results dictionary we have all we need/want so
            # the final runtime should be taken here (writing to the file should
            # not be counted):
            final_time = time.perf_counter() - initial_time

            # Note, however, that since the coherence time waits until the code
            # returns to the quantum module, the writing up of the resources to
            # a file is playing a role. Therefore, for the benchmarking, and
            # time evaluation it might make sense to avoid writting paths into
            # the output file.
            # Writing things into the `resources file` still makes sense because
            # we need to characterize the resources used. Plus, that is done
            # only at the end of the computation and thus plays no role in the
            # coherence time estimation (or any others.)
            if i < paths_to_file:
                # Writing things to file:
                file_object.write(f'\n* Shot number: {i}\n')
                if len(all_Ps) - q_count != nr_mmts:
                    val1 = len(all_Ps) - q_count
                    file_object.write(
                        f'ERROR: Nr of Paulis ({val1}) different from t+m!')
                    raise ValueError

                else:
                    file_object.write(
                        '\n The general Pauli-based computation is:\n')
                    j = q_count
                    for Pauli in all_Ps[q_count:]:
                        file_object.write(
                            f'{Pauli.return_Pauli_str()} | {all_outcs[j]}\n')
                        j += 1

                if len(pipc_Ps) - q_count > t_count:
                    val2 = len(pipc_Ps) - q_count
                    file_object.write(
                        f'ERROR: Nr of PIPC Paulis ({val2}) is larger than t!')
                    raise ValueError

                else:
                    file_object.write(
                        '\n The actual quantum computation is:\n')
                    j = q_count
                    for Pauli in pipc_Ps[q_count:]:
                        string = Pauli.return_Pauli_str(
                        ) + ' | ' + Pauli.return_Pauli_str(
                        )[0] + Pauli.return_Pauli_str(
                        )[q_count + 1:] + f' | {pipc_outcs[j]} \n'
                        file_object.write(string)
                        j += 1

                file_object.write(
                    '\n The sequence of computational outcomes is:\n')
                j = q_count + t_count
                for Pauli in all_Ps[(nr_mmts - t_count) * -1:]:
                    file_object.write(
                        f'{Pauli.return_Pauli_str()} | {all_outcs[j]}\n')
                    j += 1
                file_object.write(
                    f'\n ----------------- End of shot {i} ----------------- \n'
                )

            if 'cx' in quant_ops.keys():
                cx_count_cc.append(quant_ops['cx'])
            if ('h' in quant_ops.keys() and 's' in quant_ops.keys()):
                hs_count_cc.append(quant_ops['h'] - t_count + quant_ops['s'])
            if ('h' in quant_ops.keys() and not quant_ops.get('s')):
                hs_count_cc.append(quant_ops['h'] - t_count)

            qu_time.append(quant_time)
            cl_time.append(final_time - quant_time)
            qc_depth.append(quant_depth)

            if t_count > 1:
                min_coherence.append(min(cohe_per_shot))
                mean_coherence.append(stat.mean(cohe_per_shot))
                max_coherence.append(max(cohe_per_shot))

    with open(f'{file_loc}output/{resources_file_name}.txt',
              'w') as file_object:
        file_object.write('Resources of the original quantum circuit:\n')
        file_object.write('\n')
        file_object.write(f'Total number of qubits: {q_count}\n')
        file_object.write(f'Total number of T gates: {t_count}\n')
        file_object.write(f'Total number of CNOT gates: {cx_count}\n')
        file_object.write(f'Total number of H+S gates: {hs_count}\n')
        file_object.write(
            f'Total number of Clifford gates: {cx_count+hs_count}\n')
        file_object.write('\n')

        file_object.write(
            '----- Resources of the compiled computation: -----\n')
        file_object.write(f'Total number of qubits: {t_count}\n')
        if t_count > 1:
            file_object.write(
                f'Mean nr. of CX gates: {round(stat.mean(cx_count_cc), 3)}\n')
            file_object.write(
                f'Std CX gates: {round(stat.stdev(cx_count_cc), 3)}\n')
        file_object.write(
            f'Mean nr. of H+S gates: {round(stat.mean(hs_count_cc), 3)}\n')
        file_object.write(
            f'Std H+S gates: {round(stat.stdev(hs_count_cc), 3)}\n')
        file_object.write('\n')

        file_object.write("----- Qiskit's depth measure: -----\n")
        file_object.write(
            f'Average circuit depth: {round(stat.mean(qc_depth), 3)}\n')
        file_object.write(
            f'Std of the circuit depth: {round(stat.stdev(qc_depth), 3)}\n')
        file_object.write('\n')

        if stat.mean(cl_time) > 1:
            avg_cl_time = round(stat.mean(cl_time), 3)
            std_cl_time = round(stat.stdev(cl_time), 3)
            unit = 's/shot'
        else:
            avg_cl_time = round(stat.mean(cl_time) * 1000, 3)
            std_cl_time = round(stat.stdev(cl_time) * 1000, 3)
            unit = 'ms/shot'
        file_object.write("----- Classical runtime: -----\n")
        file_object.write(f'Average: {avg_cl_time}{unit}\n')
        file_object.write(f'Standard deviation: {std_cl_time}{unit}\n')
        file_object.write('\n')

        if stat.mean(qu_time) > 1:
            avg_qu_time = round(stat.mean(qu_time), 3)
            std_qu_time = round(stat.stdev(qu_time), 3)
            unit = 's/shot'
        else:
            avg_qu_time = round(stat.mean(qu_time) * 1000, 3)
            std_qu_time = round(stat.stdev(qu_time) * 1000, 3)
            unit = 'ms/shot'
        file_object.write("----- Quantum simulator job time: -----\n")
        file_object.write(f'Average: {avg_qu_time}{unit}\n')
        file_object.write(f'Standard deviation: {std_qu_time}{unit}\n')
        file_object.write('\n')

        file_object.write(
            "----- Coherence analysis (time in-between qu. jobs): -----\n")
        if t_count > 1:
            if stat.mean(min_coherence) > 1:
                avg_max_cohe = round(stat.mean(max_coherence), 3)
                std_max_cohe = round(stat.stdev(max_coherence), 3)
                avg_mean_cohe = round(stat.mean(mean_coherence), 3)
                std_mean_cohe = round(stat.stdev(mean_coherence), 3)
                avg_min_cohe = round(stat.mean(min_coherence), 3)
                std_min_cohe = round(stat.stdev(min_coherence), 3)
                unit = 's/shot'
            else:
                avg_max_cohe = round(stat.mean(max_coherence) * 1000, 3)
                std_max_cohe = round(stat.stdev(max_coherence) * 1000, 3)
                avg_mean_cohe = round(stat.mean(mean_coherence) * 1000, 3)
                std_mean_cohe = round(stat.stdev(mean_coherence) * 1000, 3)
                avg_min_cohe = round(stat.mean(min_coherence) * 1000, 3)
                std_min_cohe = round(stat.stdev(min_coherence) * 1000, 3)
                unit = 'ms/shot'

            file_object.write(" * Maximum time in between jobs: \n")
            file_object.write(f'Average: {avg_max_cohe}{unit}\n')
            file_object.write(f'Standard deviation: {std_max_cohe}{unit}\n')
            file_object.write('\n')

            file_object.write(" * Average time in between jobs: \n")
            file_object.write(f'Average: {avg_mean_cohe}{unit}\n')
            file_object.write(f'Standard deviation: {std_mean_cohe}{unit}\n')
            file_object.write('\n')

            file_object.write(" * Minimum time in between jobs: \n")
            file_object.write(f'Average: {avg_min_cohe}{unit}\n')
            file_object.write(f'Standard deviation: {std_min_cohe}{unit}\n')
            file_object.write('\n')

        else:
            file_object.write("!There is a single quantum measurement! \n")

        file_object.write(
            "----------------------------------------------------\n")
        file_object.write(
            f'Total classical runtime: {round(sum(cl_time), 3)}s' +
            f' ({shots} shots)\n')
        file_object.write(
            f'Total quantum simulator job time: {round(sum(qu_time), 3)}s' +
            f' ({shots} shots)\n')
        file_object.write(
            f'Total runtime: {round(sum(cl_time) + sum(qu_time), 3)}s' +
            f' ({shots} shots)\n')

    with open(f'{file_loc}output/{resources_file_name}--cx.txt',
              'w') as file_object:
        json.dump(cx_count_cc, file_object)

    with open(f'{file_loc}output/{resources_file_name}--hs.txt',
              'w') as file_object:
        json.dump(hs_count_cc, file_object)

    with open(f'{file_loc}output/{resources_file_name}--depth.txt',
              'w') as file_object:
        json.dump(qc_depth, file_object)

    with open(f'{file_loc}output/{resources_file_name}--cl_time.txt',
              'w') as file_object:
        json.dump(cl_time, file_object)

    with open(f'{file_loc}output/{resources_file_name}--qu_time.txt',
              'w') as file_object:
        json.dump(qu_time, file_object)

    with open(f'{file_loc}output/{resources_file_name}--max_cohe.txt',
              'w') as file_object:
        json.dump(max_coherence, file_object)

    with open(f'{file_loc}output/{resources_file_name}--mean_cohe.txt',
              'w') as file_object:
        json.dump(mean_coherence, file_object)

    with open(f'{file_loc}output/{resources_file_name}--min_cohe.txt',
              'w') as file_object:
        json.dump(min_coherence, file_object)

    # Saving the histogram in case we need it for later use:
    results = dict(sorted(results.items()))
    if input_file_name[-5:] == '.qasm':
        name = input_file_name[:-5]
        with open(f'{file_loc}output/Histogram_results-{name}.txt',
                  'w') as histo_file:
            json.dump(results, histo_file)
    else:
        with open(f'{file_loc}output/Histogram_results-{input_file_name}.txt',
                  'w') as histo_file:
            json.dump(results, histo_file)

    if norm_hist:
        for key, value in results.items():
            results[key] = value / shots

    if plot_hist:
        # Plotting the output distribution is such option was set to True:
        fig = plt.figure(figsize=(12, 8))
        plt.bar(*zip(*results.items()))
        plt.xlabel('Possible outcomes', fontsize=14)
        plt.xticks(fontsize=12, rotation=70)
        plt.ylabel('Probabilities', fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Output distribution', fontsize=20)
        for key, value in results.items():
            plt.text(x=key,
                     y=value,
                     s=f'{round(value, 3)}',
                     fontdict=dict(fontsize=14),
                     ha='center',
                     va='baseline')
        plt.tight_layout()
        if input_file_name[-5:] == '.qasm':
            name = input_file_name[:-5]
            fig.savefig(f'{file_loc}output/Output_distribution-{name}.pdf')
            plt.close()
        else:
            fig.savefig(
                f'{file_loc}output/Output_distribution-{input_file_name}.pdf')
            plt.close()


def hybrid_pbc(file_loc,
               input_file_name,
               clifford_file_name,
               resources_file_name,
               virtual_qubits=1,
               precision=0.1,
               confidence_level=0.99):
    """Function which carries out the hybrid Pauli-based computation to estimate
    the probability, p, that the output bit yields the outcome s=1, within an
    error of at most `precision` and using a number of virtual qubits defined by
    `virtual_qubits`.

    Args:
        file_loc (str): String indicating the location of the input file
        input_file_name (str): String with the name of the input file
        clifford_file_name (str): String with the name of the generated
        adaptive, magic-state injected Clifford circuit
        resources_file_name (str): String with the desired name for the resource
        file
        virtual_qubits (int, optional): Number of desired virtual qubits.
        Defaults to 1.
        precision (float, optional): Maximum allowed error. Defaults to 0.1.
        confidence_level (float, optional): Desired confidence level. Defaults
        to 0.99.
    """
    # Creating the folder for the output files:
    if not os.path.exists(f'{file_loc}output'):
        os.mkdir(f'{file_loc}output')

    # STEP 1 - Read in the input quantum circuit and its properties:
    qc = inp2.QuCirc(file_loc, input_file_name, virtual_qubits)
    clifford, q_count, t_count, _, _, nr_mmts = qc.msi_circuit(
        file_loc, clifford_file_name)
    # q_count -> nr. of qubits of the original quantum circuit
    # t_count -> nr. of T-gates of the original quantum circuit
    # nr_mmts -> nr. of measurements of the (new) gadgetized circuit

    for line in clifford:
        if (line.startswith('h ') or line.startswith('s ')
                or line.startswith('cx ')):
            first_gate_index = clifford.index(line)
            break

    # STEP 2 - Observe that instead of a PBC on (t_count) qubits we will have a
    # PBC on (t_count-virtual_qubits qubits) and the number of stabilizer state
    # qubits is (q_count + virtual_qubits qubits):
    t_count = t_count - virtual_qubits
    q_count = q_count + virtual_qubits

    # STEP 3 -  Determine the required number of samples:
    weights_1vq = [1 / 2, (1 - sqrt(2)) / 2, 1 / sqrt(2)]
    probs_1vq = [sqrt(2) / 4, (2 - sqrt(2)) / 4, 1 / 2]
    l1_norm = sqrt(2)**virtual_qubits
    l1_squared = 2**virtual_qubits
    fail_prob = 1 - confidence_level
    k_const = math.log(2 / fail_prob) / 2  # from Hoeffding's inequality
    nr_samples = int(math.ceil(k_const * l1_squared / precision**2))

    # STEP 4 - Perform the hybrid computation by making a weighted
    # sampling of the (smaller) PBC to be run in each iteration:
    results = []
    qu_time, cl_time, tt_time = ([], [], [])
    rand_variable = 0
    for _ in range(nr_samples):
        initial_time = time.perf_counter()

        label = ''
        for _ in range(virtual_qubits):
            chosen = str(
                np.random.choice(np.arange(0, len(weights_1vq)), p=probs_1vq))
            label += chosen

        # Finding out the signal and the unitary associated with this iteration:
        weight = 1
        for s in label:
            weight = weight * weights_1vq[int(s)]
        signal = weight / abs(weight)

        unitary = []
        for s in range(len(label)):
            string = f'{q_count - s - 1}'
            unitaries_1vq = [[['h q_stab[' + string + '];\n']],
                             [['h q_stab[' + string + '];\n'],
                              ['s q_stab[' + string + '];\n'],
                              ['s q_stab[' + string + '];\n']],
                             [['h q_stab[' + string + '];\n'],
                              ['s q_stab[' + string + '];\n']]]
            unitary.append(unitaries_1vq[int(label[s])])

        # Now that we have the correct signal and unitary for this round, we
        # can proceed with the actual hybrid computation:
        # (a) the circuit list is copied:
        clist = clifford.copy()
        # (b) we add the appropriate unitary in front of each circuit:
        for element in unitary:
            for k in range(len(element)):
                clist.insert(first_gate_index,
                             element[len(element) - k - 1][0])
        # (c) we initialize the Pauli lists
        all_Ps, all_outcs = initialize_lists(q_count, t_count)
        pipc_Ps, pipc_outcs = initialize_lists(q_count, t_count)
        mat = []

        # (d) we carry out the compilation
        fqm = False
        quant_time = 0
        for _ in range(nr_mmts):
            clist, current_Pauli = find_new_Pauli(clist, q_count, t_count)
            clist, pipc_Ps, mat, pipc_outcs, all_Ps, all_outcs, fqm = compiling(
                clist,
                q_count,
                t_count,
                first_gate_index,
                pipc_Ps,
                mat,
                pipc_outcs,
                all_Ps,
                all_outcs,
                current_Pauli,
                fqm,
                dummy=False)
            quant_time += job_time

        # The outcome is computed as:
        outcome = signal * ((-1)**all_outcs[-1]) * l1_norm
        rand_variable += (1 - outcome) / (2 * nr_samples)

        # once we have the results, we have all we need/want so that the
        # final runtime should be taken here (writing to the files should
        # not be included in this time):
        final_time = time.perf_counter() - initial_time
        tt_time.append(final_time)
        qu_time.append(quant_time)
        cl_time.append(final_time - quant_time)

        results.append(outcome)

    with open(f'{file_loc}output/{resources_file_name}--results.txt',
              'w') as file_object:
        json.dump(results, file_object)

    with open(f'{file_loc}output/{resources_file_name}--cl_time.txt',
              'w') as file_object:
        json.dump(cl_time, file_object)

    with open(f'{file_loc}output/{resources_file_name}--qu_time.txt',
              'w') as file_object:
        json.dump(qu_time, file_object)

    with open(f'{file_loc}output/{resources_file_name}--tt_time.txt',
              'w') as file_object:
        json.dump(tt_time, file_object)

    with open(f'{file_loc}output/{resources_file_name}.txt',
              'w') as file_object:
        file_object.write(f'Chosen precision = {precision}. \n')
        file_object.write(f'Chosen confidence level = {confidence_level}. \n')
        file_object.write('------------------------------------------------\n')
        file_object.write(f'The l1-norm = {l1_norm}. \n')
        file_object.write(f'Total number of samples = {nr_samples}. \n')
        file_object.write(f'The desired probability is p = {rand_variable}.\n')
        file_object.write('------------------------------------------------\n')

        if stat.mean(tt_time) < 0.01:
            file_object.write(
                f'Average time/sample = {round(stat.mean(tt_time)*1000, 1)} ms. \n'
            )
        else:
            file_object.write(
                f'Average time/sample = {round(stat.mean(tt_time), 3)} s. \n')

        if sum(tt_time) / 3600 < 1:
            file_object.write(
                f'Total time = {round(sum(tt_time)/60, 3)} minutes.')
        else:
            file_object.write(
                f'Total time = {round(sum(tt_time)/3600, 3)} hours.')
