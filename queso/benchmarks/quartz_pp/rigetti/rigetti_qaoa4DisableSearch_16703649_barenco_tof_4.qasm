OPENQASM 2.0];
include "q[elib1.inc"];
q[reg q[7];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[5], q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
rz(-pi/4) q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[3], q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
rz(pi/4) q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[5], q[6];
rx(pi) q[6];
rx(pi) q[5];
rz(pi/2) q[6];
rz(pi/2) q[5];
rx(pi/2) q[6];
rx(pi/2) q[5];
rz(-pi/2) q[6];
rz(-pi/2) q[5];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[3], q[6];
rx(pi) q[6];
cz q[3], q[5];
rz(pi/2) q[6];
rx(pi) q[5];
rx(pi/2) q[6];
rz(pi/2) q[5];
rz(-pi/2) q[6];
rx(pi/2) q[5];
rx(pi) q[6];
rz(-pi/2) q[5];
rz(pi/2) q[6];
rz(-pi/4) q[5];
rx(pi/2) q[6];
rx(pi) q[5];
rz(-pi/2) q[6];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[3], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[4], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[4], q[5];
rx(pi) q[5];
rx(pi) q[4];
rz(pi/2) q[5];
rz(pi/2) q[4];
rx(pi/2) q[5];
rx(pi/2) q[4];
rz(-pi/2) q[5];
rz(-pi/2) q[4];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
cz q[2], q[4];
rz(pi/2) q[5];
rx(pi) q[4];
rx(pi/2) q[5];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rx(pi/2) q[4];
rx(pi) q[5];
rz(-pi/2) q[4];
rz(pi/2) q[5];
rz(-pi/4) q[4];
rx(pi/2) q[5];
rx(pi) q[4];
rz(-pi/2) q[5];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[2], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[1], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rz(-pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[0], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[1], q[4];
rx(pi) q[4];
rx(pi) q[1];
rz(pi/2) q[4];
rz(pi/2) q[1];
rx(pi/2) q[4];
rx(pi/2) q[1];
rz(-pi/2) q[4];
rz(-pi/2) q[1];
rz(-pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[0], q[4];
rx(pi) q[4];
cz q[0], q[1];
rz(pi/2) q[4];
rx(pi) q[1];
rx(pi/2) q[4];
rz(pi/2) q[1];
rz(-pi/2) q[4];
rx(pi/2) q[1];
rz(pi/4) q[4];
rz(-pi/2) q[1];
rx(pi) q[4];
rx(pi) q[1];
rz(pi/2) q[4];
rz(pi/2) q[1];
rx(pi/2) q[4];
rx(pi/2) q[1];
rz(-pi/2) q[4];
rz(-pi/2) q[1];
cz q[4], q[5];
cz q[0], q[1];
rx(pi) q[5];
rx(pi) q[1];
rz(pi/2) q[5];
rz(pi/2) q[1];
rx(pi/2) q[5];
rx(pi/2) q[1];
rz(-pi/2) q[5];
rz(-pi/2) q[1];
rz(pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[4], q[5];
rx(pi) q[5];
rx(pi) q[4];
rz(pi/2) q[5];
rz(pi/2) q[4];
rx(pi/2) q[5];
rx(pi/2) q[4];
rz(-pi/2) q[5];
rz(-pi/2) q[4];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
cz q[2], q[4];
rz(pi/2) q[5];
rx(pi) q[4];
rx(pi/2) q[5];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rx(pi/2) q[4];
rx(pi) q[5];
rz(-pi/2) q[4];
rz(pi/2) q[5];
rx(pi) q[4];
rx(pi/2) q[5];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rx(pi/2) q[4];
cz q[5], q[6];
rz(-pi/2) q[4];
rx(pi) q[6];
cz q[2], q[4];
rz(pi/2) q[6];
rx(pi) q[4];
rx(pi/2) q[6];
rz(pi/2) q[4];
rz(-pi/2) q[6];
rx(pi/2) q[4];
rz(pi/4) q[6];
rz(-pi/2) q[4];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[3], q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
rz(-pi/4) q[6];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[5], q[6];
rx(pi) q[6];
rx(pi) q[5];
rz(pi/2) q[6];
rz(pi/2) q[5];
rx(pi/2) q[6];
rx(pi/2) q[5];
rz(-pi/2) q[6];
rz(-pi/2) q[5];
rx(pi) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(-pi/2) q[6];
cz q[3], q[6];
rx(pi) q[6];
cz q[3], q[5];
rz(pi/2) q[6];
rx(pi) q[5];
rx(pi/2) q[6];
rz(pi/2) q[5];
rz(-pi/2) q[6];
rx(pi/2) q[5];
rx(pi) q[6];
rz(-pi/2) q[5];
rz(pi/2) q[6];
rz(pi/4) q[5];
rx(pi/2) q[6];
rx(pi) q[5];
rz(-pi/2) q[6];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[3], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[4], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[4], q[5];
rx(pi) q[5];
rx(pi) q[4];
rz(pi/2) q[5];
rz(pi/2) q[4];
rx(pi/2) q[5];
rx(pi/2) q[4];
rz(-pi/2) q[5];
rz(-pi/2) q[4];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
cz q[2], q[4];
rz(pi/2) q[5];
rx(pi) q[4];
rx(pi/2) q[5];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rx(pi/2) q[4];
rx(pi) q[5];
rz(-pi/2) q[4];
rz(pi/2) q[5];
rx(pi) q[4];
rx(pi/2) q[5];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[2], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[1], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[0], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rz(-pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[1], q[4];
rx(pi) q[4];
rx(pi) q[1];
rz(pi/2) q[4];
rz(pi/2) q[1];
rx(pi/2) q[4];
rx(pi/2) q[1];
rz(-pi/2) q[4];
rz(-pi/2) q[1];
rz(pi/4) q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[0], q[4];
rx(pi) q[4];
cz q[0], q[1];
rz(pi/2) q[4];
rx(pi) q[1];
rx(pi/2) q[4];
rz(pi/2) q[1];
rz(-pi/2) q[4];
rx(pi/2) q[1];
rz(-pi/4) q[4];
rz(-pi/2) q[1];
rx(pi) q[4];
rx(pi) q[1];
rz(pi/2) q[4];
rz(pi/2) q[1];
rx(pi/2) q[4];
rx(pi/2) q[1];
rz(-pi/2) q[4];
rz(-pi/2) q[1];
cz q[4], q[5];
cz q[0], q[1];
rx(pi) q[5];
rx(pi) q[1];
rz(pi/2) q[5];
rz(pi/2) q[1];
rx(pi/2) q[5];
rx(pi/2) q[1];
rz(-pi/2) q[5];
rz(-pi/2) q[1];
rz(pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/4) q[5];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[4], q[5];
rx(pi) q[5];
rx(pi) q[4];
rz(pi/2) q[5];
rz(pi/2) q[4];
rx(pi/2) q[5];
rx(pi/2) q[4];
rz(-pi/2) q[5];
rz(-pi/2) q[4];
rx(pi) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(-pi/2) q[5];
cz q[2], q[5];
rx(pi) q[5];
cz q[2], q[4];
rz(pi/2) q[5];
rx(pi) q[4];
rx(pi/2) q[5];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rx(pi/2) q[4];
rx(pi) q[5];
rz(-pi/2) q[4];
rz(pi/2) q[5];
rz(pi/4) q[4];
rx(pi/2) q[5];
rx(pi) q[4];
rz(-pi/2) q[5];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
cz q[2], q[4];
rx(pi) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(-pi/2) q[4];
rz(-pi/4) q[4];