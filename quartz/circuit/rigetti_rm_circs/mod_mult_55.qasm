OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
x q[2];
x q[3];
x q[4];
x q[5];
cz q[2],q[7];
rz(pi*0.500000) q[3];
rz(pi*0.500000) q[4];
rz(pi*0.500000) q[5];
x q[7];
rx1 q[3];
rx1 q[4];
rx1 q[5];
rz(pi*0.500000) q[7];
rz(pi*-0.500000) q[3];
rz(pi*-0.500000) q[4];
rz(pi*-0.500000) q[5];
rx1 q[7];
rz(pi*-0.500000) q[7];
rz(pi*-0.250000) q[7];
x q[7];
rz(pi*0.500000) q[7];
rx1 q[7];
rz(pi*-0.500000) q[7];
cz q[0],q[7];
x q[7];
rz(pi*0.500000) q[7];
rx1 q[7];
rz(pi*-0.500000) q[7];
rz(pi*0.250000) q[7];
x q[7];
rz(pi*0.500000) q[7];
rx1 q[7];
rz(pi*-0.500000) q[7];
cz q[2],q[7];
x q[2];
x q[7];
cz q[2],q[8];
rz(pi*0.500000) q[7];
x q[8];
rx1 q[7];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[7];
rx1 q[8];
rz(pi*-0.250000) q[7];
rz(pi*-0.500000) q[8];
x q[7];
rz(pi*0.250000) q[8];
rz(pi*0.500000) q[7];
x q[8];
rx1 q[7];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[7];
rx1 q[8];
cz q[0],q[7];
rz(pi*-0.500000) q[8];
x q[0];
x q[7];
cz q[0],q[8];
rz(pi*0.500000) q[7];
x q[8];
rx1 q[7];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[7];
rx1 q[8];
rz(pi*0.250000) q[7];
rz(pi*-0.500000) q[8];
x q[7];
rz(pi*-0.250000) q[8];
rz(pi*0.500000) q[7];
x q[8];
rx1 q[7];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[7];
rx1 q[8];
cz q[7],q[6];
rz(pi*-0.500000) q[8];
x q[6];
cz q[2],q[8];
rz(pi*0.500000) q[6];
x q[8];
rx1 q[6];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[6];
rx1 q[8];
rz(pi*-0.250000) q[6];
rz(pi*-0.500000) q[8];
x q[6];
rz(pi*0.250000) q[8];
rz(pi*0.500000) q[6];
x q[8];
rx1 q[6];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[6];
rx1 q[8];
cz q[1],q[6];
rz(pi*-0.500000) q[8];
x q[6];
cz q[0],q[8];
rz(pi*0.500000) q[6];
x q[8];
rx1 q[6];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[6];
rx1 q[8];
rz(pi*0.250000) q[6];
rz(pi*-0.500000) q[8];
x q[6];
rz(pi*-0.250000) q[8];
rz(pi*0.500000) q[6];
x q[8];
rx1 q[6];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[6];
rx1 q[8];
cz q[7],q[6];
rz(pi*-0.500000) q[8];
x q[6];
x q[7];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[7];
rx1 q[6];
rx1 q[7];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[7];
rz(pi*-0.250000) q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
cz q[1],q[6];
cz q[1],q[7];
x q[6];
x q[7];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[7];
rx1 q[6];
rx1 q[7];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[7];
rz(pi*0.250000) q[6];
rz(pi*-0.250000) q[7];
x q[6];
x q[7];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[7];
rx1 q[6];
rx1 q[7];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[7];
cz q[6],q[5];
cz q[1],q[7];
cz q[6],q[3];
x q[5];
x q[1];
x q[7];
x q[3];
rz(pi*0.500000) q[5];
rz(pi*0.500000) q[7];
rz(pi*0.500000) q[3];
rx1 q[5];
rx1 q[7];
rx1 q[3];
rz(pi*-0.500000) q[5];
rz(pi*-0.500000) q[7];
rz(pi*-0.500000) q[3];
rz(pi*0.250000) q[7];
x q[7];
rz(pi*0.500000) q[7];
rx1 q[7];
rz(pi*-0.500000) q[7];
cz q[8],q[7];
cz q[8],q[6];
x q[7];
x q[6];
rz(pi*0.500000) q[7];
rz(pi*0.500000) q[6];
rx1 q[7];
rx1 q[6];
rz(pi*-0.500000) q[7];
rz(pi*-0.500000) q[6];
x q[7];
rz(pi*0.250000) q[6];
cz q[7],q[3];
x q[6];
x q[3];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[3];
rx1 q[6];
rx1 q[3];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[3];
cz q[1],q[6];
rz(pi*-0.250000) q[3];
x q[6];
x q[3];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[3];
rx1 q[6];
rx1 q[3];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[3];
rz(pi*-0.250000) q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
cz q[8],q[6];
x q[6];
x q[8];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[8];
rx1 q[6];
rx1 q[8];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[8];
rz(pi*0.250000) q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
cz q[1],q[6];
cz q[1],q[8];
x q[6];
x q[8];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[8];
rx1 q[6];
rx1 q[8];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[8];
rz(pi*-0.250000) q[6];
rz(pi*0.250000) q[8];
x q[6];
x q[8];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[8];
rx1 q[6];
rx1 q[8];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[8];
cz q[6],q[4];
cz q[1],q[8];
x q[4];
cz q[1],q[3];
x q[8];
rz(pi*0.500000) q[4];
x q[3];
rz(pi*0.500000) q[8];
rx1 q[4];
rz(pi*0.500000) q[3];
rx1 q[8];
rz(pi*-0.500000) q[4];
rx1 q[3];
rz(pi*-0.500000) q[8];
rz(pi*-0.500000) q[3];
rz(pi*-0.250000) q[8];
rz(pi*0.250000) q[3];
x q[8];
x q[3];
rz(pi*0.500000) q[8];
rz(pi*0.500000) q[3];
rx1 q[8];
rx1 q[3];
rz(pi*-0.500000) q[8];
rz(pi*-0.500000) q[3];
cz q[5],q[8];
cz q[7],q[3];
x q[8];
cz q[7],q[5];
x q[3];
rz(pi*0.500000) q[8];
x q[5];
rz(pi*0.500000) q[3];
rx1 q[8];
rz(pi*0.500000) q[5];
rx1 q[3];
rz(pi*-0.500000) q[8];
rx1 q[5];
rz(pi*-0.500000) q[3];
cz q[2],q[8];
rz(pi*-0.500000) q[5];
rz(pi*-0.250000) q[3];
x q[8];
rz(pi*0.250000) q[5];
x q[3];
rz(pi*0.500000) q[8];
x q[5];
rz(pi*0.500000) q[3];
rx1 q[8];
rz(pi*0.500000) q[5];
rx1 q[3];
rz(pi*-0.500000) q[8];
rx1 q[5];
rz(pi*-0.500000) q[3];
rz(pi*-0.250000) q[8];
rz(pi*-0.500000) q[5];
cz q[1],q[3];
x q[8];
x q[1];
x q[3];
rz(pi*0.500000) q[8];
cz q[1],q[5];
rz(pi*0.500000) q[3];
rx1 q[8];
x q[5];
rx1 q[3];
rz(pi*-0.500000) q[8];
rz(pi*0.500000) q[5];
rz(pi*-0.500000) q[3];
cz q[0],q[8];
rx1 q[5];
rz(pi*0.250000) q[3];
x q[8];
rz(pi*-0.500000) q[5];
x q[3];
rz(pi*0.500000) q[8];
rz(pi*-0.250000) q[5];
rz(pi*0.500000) q[3];
rx1 q[8];
x q[5];
rx1 q[3];
rz(pi*-0.500000) q[8];
rz(pi*0.500000) q[5];
rz(pi*-0.500000) q[3];
rz(pi*0.250000) q[8];
rx1 q[5];
x q[8];
rz(pi*-0.500000) q[5];
rz(pi*0.500000) q[8];
cz q[7],q[5];
rx1 q[8];
x q[7];
x q[5];
rz(pi*-0.500000) q[8];
rz(pi*0.500000) q[5];
cz q[2],q[8];
rx1 q[5];
x q[8];
x q[2];
rz(pi*-0.500000) q[5];
rz(pi*0.500000) q[8];
rz(pi*0.500000) q[2];
rz(pi*0.250000) q[5];
rx1 q[8];
rx1 q[2];
x q[5];
rz(pi*-0.500000) q[8];
rz(pi*-0.500000) q[2];
rz(pi*0.500000) q[5];
rz(pi*-0.250000) q[8];
rx1 q[5];
x q[8];
rz(pi*-0.500000) q[5];
rz(pi*0.500000) q[8];
cz q[1],q[5];
rx1 q[8];
x q[5];
rz(pi*-0.500000) q[8];
rz(pi*0.500000) q[5];
cz q[0],q[8];
rx1 q[5];
cz q[0],q[2];
x q[8];
rz(pi*-0.500000) q[5];
x q[2];
rz(pi*0.500000) q[8];
rz(pi*-0.250000) q[5];
rz(pi*0.500000) q[2];
rx1 q[8];
x q[5];
rx1 q[2];
rz(pi*-0.500000) q[8];
rz(pi*0.500000) q[5];
rz(pi*-0.500000) q[2];
rz(pi*0.250000) q[8];
rx1 q[5];
rz(pi*-0.250000) q[2];
rz(pi*-0.500000) q[5];
x q[2];
cz q[5],q[8];
rz(pi*0.500000) q[2];
x q[8];
rx1 q[2];
rz(pi*0.500000) q[8];
rz(pi*-0.500000) q[2];
rx1 q[8];
cz q[0],q[2];
rz(pi*-0.500000) q[8];
rz(pi*0.250000) q[0];
x q[2];
x q[0];
rz(pi*0.500000) q[2];
rx1 q[2];
rz(pi*-0.500000) q[2];
rz(pi*0.250000) q[2];