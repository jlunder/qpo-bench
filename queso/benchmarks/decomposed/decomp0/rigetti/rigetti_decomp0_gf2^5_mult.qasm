OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[6], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[4], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[6], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[4], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[4], q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[4], q[6];
rz(pi/4) q[4];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cz q[7], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[3], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[7], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[3], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[3], q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[3], q[7];
rz(pi/4) q[3];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cz q[8], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[2], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[8], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[2], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[2], q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[2], q[8];
rz(pi/4) q[2];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cz q[9], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[1], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[9], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[1], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[1], q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[1], q[9];
rz(pi/4) q[1];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[7], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[4], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[7], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[4], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[4], q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[4], q[7];
rz(pi/4) q[4];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cz q[8], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[3], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[8], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[3], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[3], q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[3], q[8];
rz(pi/4) q[3];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cz q[9], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[2], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[9], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[2], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[2], q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[2], q[9];
rz(pi/4) q[2];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[8], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[4], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[8], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[4], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[4], q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[4], q[8];
rz(pi/4) q[4];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cz q[9], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[3], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[9], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[3], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[3], q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[3], q[9];
rz(pi/4) q[3];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[9], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[4], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[9], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[4], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[13], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[11], q[13];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[4], q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[4], q[9];
rz(pi/4) q[4];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[12], q[14];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[10], q[12];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[5], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[4], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[5], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[4], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[4], q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[4], q[5];
rz(pi/4) q[4];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cz q[5], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[6], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[3], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[6], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[3], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[3], q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[3], q[6];
rz(pi/4) q[3];
cz q[3], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[5], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[3], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[3], q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[3], q[5];
rz(pi/4) q[3];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cz q[5], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cz q[6], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[7], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[2], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[7], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[2], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[2], q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[2], q[7];
rz(pi/4) q[2];
cz q[2], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[6], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[2], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[2], q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[2], q[6];
rz(pi/4) q[2];
cz q[2], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[5], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[2], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[2], q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[2], q[5];
rz(pi/4) q[2];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cz q[5], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cz q[6], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cz q[7], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[8], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[1], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[8], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[1], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[1], q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[1], q[8];
rz(pi/4) q[1];
cz q[1], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[7], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[1], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[1], q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[1], q[7];
rz(pi/4) q[1];
cz q[1], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[6], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[1], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[1], q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[1], q[6];
rz(pi/4) q[1];
cz q[1], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[5], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[1], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[1], q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[1], q[5];
rz(pi/4) q[1];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cz q[5], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cz q[6], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cz q[7], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cz q[8], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[9], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[0], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[9], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cz q[0], q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[0], q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[0], q[9];
rz(pi/4) q[0];
cz q[0], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[8], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
cz q[0], q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[0], q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[0], q[8];
rz(pi/4) q[0];
cz q[0], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[7], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[0], q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[0], q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[0], q[7];
rz(pi/4) q[0];
cz q[0], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[6], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[0], q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[0], q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[0], q[6];
rz(pi/4) q[0];
cz q[0], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[5], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[0], q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[0], q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[0], q[5];
rz(pi/4) q[0];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];