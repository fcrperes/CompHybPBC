OPENQASM 2.0;
include "qelib1.inc";
qreg q_stab[4];
qreg q_magic[6];
creg c_stab[2];
creg c_magic[6];
h q_stab[2];
cx q_stab[1],q_stab[2];
cx q_stab[2],q_stab[3];
measure q_stab[3] -> c_stab[1];
if(c_stab[1]==1) s q_stab[2];
s q_stab[2];
s q_stab[2];
s q_stab[2];
cx q_stab[0],q_stab[2];
cx q_stab[2],q_magic[0];
measure q_magic[0] -> c_magic[0];
if(c_magic[0]==1) s q_stab[2];
cx q_stab[1],q_stab[2];
cx q_stab[2],q_magic[1];
measure q_magic[1] -> c_magic[1];
if(c_magic[1]==1) s q_stab[2];
s q_stab[2];
s q_stab[2];
s q_stab[2];
cx q_stab[0],q_stab[2];
cx q_stab[2],q_magic[2];
measure q_magic[2] -> c_magic[2];
if(c_magic[2]==1) s q_stab[2];
cx q_stab[1],q_magic[3];
measure q_magic[3] -> c_magic[3];
if(c_magic[3]==1) s q_stab[1];
h q_stab[2];
cx q_stab[0],q_stab[1];
cx q_stab[1],q_magic[4];
measure q_magic[4] -> c_magic[4];
if(c_magic[4]==1) s q_stab[1];
s q_stab[1];
s q_stab[1];
s q_stab[1];
cx q_stab[0],q_magic[5];
measure q_magic[5] -> c_magic[5];
if(c_magic[5]==1) s q_stab[0];
cx q_stab[0],q_stab[1];
measure q_stab[2] -> c_stab[0];
