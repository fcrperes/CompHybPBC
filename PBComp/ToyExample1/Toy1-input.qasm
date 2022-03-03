OPENQASM 2.0;
include "qelib1.inc";
qreg q666[1];
creg c98[1];
h q666[0];
t q666[0];
h q666[0];
measure q666[0] -> c98[0];