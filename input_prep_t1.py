'''
--------------------------------------------------------------------------------
This module contains:

* Error classes: which characterize the different errors that can be raised if
the input .qasm quantum circuit exhibits invalid properties.

* The QuCirc class: which describes a general QuCirc object characterized by the
location and name of the input .qasm file(`input_file_loc` & `input_file_name`).
This means that an instance of this class (i.e. a specific quantum circuit) is
defined by the appropriate .qasm file. Once such an instance has been defined,
the methods within the class can be applied to it.

These methods are the following:

* Method 1: get_circuit_data(self)
            - This method reads the .qasm input file selected by the user
            - Then it formats the file, removing blank and commented lines
            - It returns a list with all the relevant information in the input
            file, i.e., all the relevant information about the quantum circuit

            - Summary: this method returns all the relevant information about
            the quantum circuit instance to which it is applied.

* Method 2: verify_input_circuit(self)
            - This method uses the information list produced by the previous
            method and verifies whether the input quantum circuit has a valid
            form. Namely the circuit must:
                * have 1 qreg and 1 creg (a total of exactly 2 registers);
                * have at least 1 measurement;
                * be a Clifford+T circuit (other unitaries are not allowed);
                * have at least 1 T gate, otherwise it is a simple stabilizer
                circuit (efficiently classically simulatable).
            - This method might raise several different errors depending on
            whichever of the above conditions is/are not met
            - If the circuit is valid, the method edits the list, changing the
            names of the quantum and classical registers to `q_stab` & `c_stab`,
            respectively. Then, the edited list, the number of qubits, the
            T-count & the total number of measurements are returned (all
            important for later steps of the compilation/computation procedure)

            - Summary: this method either warns the user that the input quantum
            circuit instance is invalid or else (if the circuit is valid) it
            returns the properties of that instance.

* Method 3: msi_circuit(self, msi_clifford_file_loc, msi_clifford_file_name)
            - This method converts the Clifford+T circuit instance into a
            gadgetized (adaptive) Clifford circuit which is equivalent to the
            original one. This adaptive Clifford circuit with magic state
            injection is exported to a different .qasm file where:
                * an auxiliary quantum register is created with the number of
                qubits = to the T-count and entitled `q_magic`;
                * an auxiliary classical register (`c_magic`) is created to
                store the outcomes of the measurements of the auxiliary (magic)
                qubits
                * each T gate acting on a qubit "q_stab[i]" is replaced by a
                T-gadget, written in terms of the conditional "IF" statement;
                * we know that the qubits in the auxiliary register must be in
                state |A>, but we do not do this explicitly because it is not
                convenient for the next stage of the compilation procedure. It
                should simply be kept in the back of our mind that the qubits in
                `q_magic` are magic-state qubits.

This class is helpful because it can be used to define different quantum circuit
instances, and then provides methods which work through those instances to turn
them into valid quantum circuits for the next stages of the computation (encoded
in other modules).


Author: F.C.R. Peres
Creation date: 02/06/2021
Last updated: 28/02/2022
--------------------------------------------------------------------------------
'''


class Error(Exception):
    """Base class for other exceptions"""
    pass


class RegisterInputError(Error):
    """Exception raised for errors associated with the registers of the input
    quantum circuit.
    """
    def __init__(
        self,
        message="Invalid circuit form: registers are not correctly specified!"
    ):
        self.message = message
        super().__init__(self.message)


class MeasurementInputError(Error):
    """Exception raised when there are no measurements on the input quantum
    circuit.
    """
    def __init__(self,
                 message="Invalid circuit form: circuit has no measurements!"):
        self.message = message
        super().__init__(self.message)


class StabilizerInputError(Error):
    """Exception raised when the input quantum circuit is a stabilizer
    circuit.
    """
    def __init__(
            self,
            message="Compilation is unnecessary: input is a stabilizer circuit."
    ):
        self.message = message
        super().__init__(self.message)


class NonCliffordTGenInputError(Error):
    """Exception raised when the input circuit has operations other than the
    ones from the {H, S, CNOT, T} generating set.
    """
    def __init__(
            self,
            message="Invalid circuit form: illegal operations are being used!"
    ):
        self.message = message
        super().__init__(self.message)


class QuCirc():
    """ This class represents a given Clifford+T input quantum circuit"""
    def __init__(self, input_file_loc, input_file_name):
        """Initialize the attributes that are used to describe the input quantum
        circuit

        Args:
            input_file_loc (str): Location of the .qasm input file
            input_file_name (str): Name of the .qasm input file
        """
        # defining the two attributes of the input quantum circuits:
        self.input_file_loc = input_file_loc
        if input_file_name[-5:] == '.qasm':
            self.input_file_name = input_file_name
        else:
            self.input_file_name = input_file_name + '.qasm'

        # defining other gen. variables useful within the methods of this class:
        self.qasm_header = ['OPENQASM ', 'include ']
        self.qasm_registers = ['qreg ', 'creg ']
        self.qasm_allowed_operations = [
            'h ', 's ', 'cx ', 't ', 'measure ', 'barrier '
        ]
        self.qasm_all = [
            'OPENQASM ', 'include ', 'qreg ', 'creg ', 'h ', 's ', 'cx ', 't ',
            'measure ', 'barrier '
        ]

    def get_circuit_data(self):
        """This method is applied to an instantiated quantum circuit, edits it
        and returns a list with the relevant information.

        Returns:
            list (str): List with all the relevant information about the quantum
            circuit; the elements of this list are the non-blank, non-commented
            lines from the input .qasm file.
        """
        with open(
                f'{self.input_file_loc}{self.input_file_name}') as file_object:
            file_lines = list(file_object)

        for line in list(file_lines):
            if line == '\n':
                file_lines.remove(line)
            elif line.startswith('//'):
                file_lines.remove(line)

        return file_lines

    def verify_input_circuit(self):
        """This method evaluates whether or not the quantum circuit instance
        encodes a valid input quantum circuit and edits it further, changing the
        names of the quantum and classical registers respectively to `q_stab` &
        `c_stab`.

        Raises:
            NonCliffordTGenInputError: Exception raised if disallowed operations
            are being used
            StabilizerInputError: Exception raised if the input quantum circuit
            is a stabilizer circuit
            MeasurementInputError: Exception raised if there are no measurements
            in the input quantum circuit
            RegisterInputError: Exception raised if the nr of quantum registers
            differs from 1 or the total nr of registers differs from 2
        
        Returns:
            tuple (list[str], int, int, int): If the quantum circuit intance is
            valid, we get an updated/edited list describing it, and 3 properties
            of that circuit: 1st the number of qubits, 2nd its T-count, and 3rd
            the number of measurements.
        """
        t_count = 0  # number of T gates in the quantum circuit (must be >0)
        mmt_count = 0  # number of measurements in the circuit (must be >0)
        nr_qregs = 0  # number of quantum registers in the circuit (must be =1)
        circuit_header, circuit_registers, circuit_operations = [[], [], []]

        circuit_list = self.get_circuit_data()

        for line in circuit_list:
            for element in self.qasm_all:
                if line.startswith(element):
                    if element in self.qasm_header:
                        circuit_header.append(line)
                    elif element in self.qasm_registers:
                        circuit_registers.append(line)
                    elif element in self.qasm_allowed_operations:
                        circuit_operations.append(line)

            if line.startswith('t '):
                t_count += 1
            elif line.startswith('measure '):
                mmt_count += 1
            elif line.startswith('qreg '):
                x = line.partition('[')
                y = x[2].partition(']')
                q_count = int(y[0])
                nr_qregs += 1

        if not (len(circuit_list) == len(circuit_header) +
                len(circuit_registers) + len(circuit_operations)):
            raise NonCliffordTGenInputError()
        elif not t_count > 0:
            raise StabilizerInputError()
        elif not mmt_count > 0:
            raise MeasurementInputError()
        elif not (len(circuit_registers) == 2 and nr_qregs == 1):
            raise RegisterInputError()

        # the next bit of code guarantees that the quantum register is named
        # `q_stab` and that the classical register is named `c_stab`:
        it = 0
        for line in circuit_list:
            if line.startswith('qreg '):
                x = line.partition('[')
                y = x[0].partition(' ')
                qreg_name = y[2]
                it += 1
            elif line.startswith('creg '):
                x = line.partition('[')
                y = x[0].partition(' ')
                creg_name = y[2]
                it += 1
            else:
                if it == 2:
                    break

        it = 0
        while it < len(circuit_list):
            circuit_list[it] = circuit_list[it].replace(
                f'{qreg_name}[', 'q_stab[')
            circuit_list[it] = circuit_list[it].replace(
                f'{creg_name}[', 'c_stab[')
            it += 1

        return circuit_list, q_count, t_count, mmt_count

    def msi_circuit(self, msi_clifford_file_loc, msi_clifford_file_name):
        """This method takes in a certain quantum circuit instance and creates
        a new .qasm file with the gadgetized (i.e., adaptive) Clifford quantum
        circuit to be used in the compilation. An important observation is that
        this method creates a second qubit register (which holds the auxiliary
        qubits). We must remember that these need to be initialized in the magic
        state |A>, even though this method does not do this because it is not
        convenient for the next stages of the computation.

        Args:
            msi_clifford_file_loc (str): Location for the output gadgetized
            Clifford .qasm file
            msi_clifford_file_name (str): Name for the output gadgetized
            Clifford .qasm file

        Returns:
            tuple (list[str], int, int, int, int, int): Tuple with the list of
            operations associated with the adaptive Clifford circuit with magic
            state injection, 4 properties of the original circuit: (1) the
            number of qubits, (2) its T-count, (3) the number of controlled-NOT
            gates, and (4) the number of single-qubit Clifford gates (H + S),
            and finally the number of measurements of the (new) gadgetized
            quantum circuit.
        """
        cx_count = 0  # number of CNOT gates in the original quantum circuit
        hs_count = 0  # number of single-qubit Clifford gates in the circuit

        circuit_list, q_count, t_count, mmt_count = self.verify_input_circuit()

        for line in circuit_list:
            if line.startswith('cx '):
                cx_count += 1
            elif line.startswith('h ') or line.startswith('s '):
                hs_count += 1

        for line in circuit_list:
            if line.startswith('creg '):
                index = circuit_list.index(line)
                circuit_list.insert(index, f'qreg q_magic[{t_count}];\n')
                circuit_list.insert(index + 2, f'creg c_magic[{t_count}];\n')
                break

        k = 0
        for line in circuit_list:
            if line.startswith('t '):
                index = circuit_list.index(line)

                x = line.partition('[')
                y = x[2].partition(']')
                i = y[0]  # qubit to which the T gate is being applied

                line = line.replace('t ', 'cx ')
                circuit_list[index] = line.replace(';', f',q_magic[{k}];')
                circuit_list.insert(
                    index + 1, f'measure q_magic[{k}] -> c_magic[{k}];\n')
                circuit_list.insert(index + 2,
                                    f'if(c_magic[{k}]==1) s q_stab[{i}];\n')
                k += 1

        if not msi_clifford_file_name[-5:] == '.qasm':
            msi_clifford_file_name += '.qasm'

        with open(f'{msi_clifford_file_loc}{msi_clifford_file_name}',
                  'w') as file_object:
            for line in circuit_list:
                file_object.write(line)

        nr_mmts = mmt_count + t_count
        return circuit_list, q_count, t_count, cx_count, hs_count, nr_mmts
