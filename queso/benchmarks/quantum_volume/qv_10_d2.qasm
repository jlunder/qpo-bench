OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
rz(0.519935195846505) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(4.35675030672066) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(11.0192375793326) q[0];
rz(0.218287194703077) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(4.72024624159304) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(7.67980850866178) q[1];
rz(-1.77739273916661) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5.39668821975096) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(8.11298412444554) q[2];
rz(2.56960487925017) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(4.71185136134313) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(11.203963869718) q[3];
rz(-2.30341018735131) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(6.03516905513309) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(7.78063200224338) q[4];
cx q[0],q[4];
rz(-pi/2) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(4.2591049419817) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(5*pi/2) q[0];
rz(-0.471887839081803) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(5.1390356678092) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(6.96451744496838) q[4];
cx q[0],q[4];
rz(-pi/2) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(3.20142190731247) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(3*pi) q[0];
rz(-0.615479708670387) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(5*pi/3) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(10.0402576694398) q[4];
cx q[0],q[4];
rz(0.509018659758103) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(4.67817888975634) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(7.4569261020614) q[0];
rz(-1.33950888270632) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(4.6119189520155) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(6.41395278998916) q[0];
rz(0.305264009842789) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(4.71877363365538) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(11.089438454863) q[4];
rz(-2.49377319275428) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(5.94779130465807) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(7.69430860999491) q[4];
rz(0.492001589453772) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(5.20171479613124) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(7.61648759940815) q[5];
rz(0.954034155514282) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(4.24909912165346) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(9.38815784892292) q[6];
cx q[6],q[5];
rz(-0.246381070305069) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(4.95161442818976) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(7.03882329181164) q[5];
rz(-pi/2) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(3.87267389316291) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(5*pi/2) q[6];
cx q[6],q[5];
rz(-0.615479708670387) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(5*pi/3) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(10.0402576694398) q[5];
rz(-pi/2) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(3.50261699423629) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(3*pi) q[6];
cx q[6],q[5];
rz(-3.0775423748696) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(4.10055014208502) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(10.0278304591625) q[5];
rz(-0.403093850466698) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(4.72077690296521) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(9.87650869981741) q[5];
rz(-1.28844100080881) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(3.79280713940189) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(7.38372036258445) q[6];
rz(-2.79936409334455) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(5.73636659575073) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(6.8967600001688) q[6];
cx q[4],q[6];
rz(-pi/2) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(4.08953301324418) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(5*pi/2) q[4];
rz(-0.406992452212986) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(5.08931200662161) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(6.98991093779035) q[6];
cx q[4],q[6];
rz(pi/2) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(4.03102506196206) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(2*pi) q[4];
rz(-0.615479708670387) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(5*pi/3) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(10.0402576694398) q[6];
cx q[4],q[6];
rz(2.33788667835314) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(4.87173931749261) q[4];
rz(-pi/2) q[4];
h q[4];
rz(-pi/2) q[4];
rz(9.12094541134197) q[4];
rz(0.865871868360644) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(4.82200474711661) q[6];
rz(-pi/2) q[6];
h q[6];
rz(-pi/2) q[6];
rz(12.30909188991) q[6];
rz(-0.918782977914632) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(5.9825140480638) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(7.59973078410625) q[7];
cx q[1],q[7];
rz(-pi/2) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(3.58653887990116) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(5*pi/2) q[1];
rz(-0.149758851978713) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(4.86049613497075) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(7.05745227769667) q[7];
cx q[1],q[7];
rz(-pi/2) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(3.41030278733719) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(3*pi) q[1];
rz(-0.615479708670387) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(5*pi/3) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(10.0402576694398) q[7];
cx q[1],q[7];
rz(-1.71930312726728) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(4.11542576258155) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(7.42449589596788) q[1];
rz(2.27688363996349) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(5.53670439548377) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(6.44065299406012) q[1];
rz(1.71467371949005) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(5.13427428210668) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(12.1197113779499) q[7];
rz(-1.23589709568073) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(4.37284520176765) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(11.6154630034237) q[7];
cx q[7],q[0];
rz(-0.580763586577355) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(5.21420473123886) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(6.91569690907415) q[0];
rz(-pi/2) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(4.01552197822169) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(5*pi/2) q[7];
cx q[7],q[0];
rz(-0.615479708670387) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(5*pi/3) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(10.0402576694398) q[0];
rz(pi/2) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(3.73730190694467) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(2*pi) q[7];
cx q[7],q[0];
rz(-1.32271549672067) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(5.10255480736031) q[0];
rz(-pi/2) q[0];
h q[0];
rz(-pi/2) q[0];
rz(11.554163001161) q[0];
rz(0.348138456107168) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(5.99594826566058) q[7];
rz(-pi/2) q[7];
h q[7];
rz(-pi/2) q[7];
rz(12.3552382416281) q[7];
rz(-2.7393683729959) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(4.61896358182315) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(7.32712196558492) q[8];
cx q[8],q[2];
rz(-0.497188288563362) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5.15743213929586) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(6.95383526998043) q[2];
rz(-pi/2) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(4.22362825441885) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(5*pi/2) q[8];
cx q[8],q[2];
rz(-0.615479708670387) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5*pi/3) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(10.0402576694398) q[2];
rz(-pi/2) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(3.34774477878305) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(3*pi) q[8];
cx q[8],q[2];
rz(-1.7821656764619) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5.2290108852502) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(9.60121763872499) q[2];
rz(2.51331676284677) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5.51022521269108) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(8.33681518808107) q[2];
rz(0.595818507171779) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(4.111392842778) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(12.1636974002886) q[8];
rz(1.76907971783162) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(4.59900301603629) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(12.2676833933975) q[8];
cx q[8],q[1];
rz(-0.30571691030935) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(5.00474179122275) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(7.023227740643) q[1];
rz(-pi/2) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(4.22423357232692) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(5*pi/2) q[8];
cx q[8],q[1];
rz(-0.615479708670387) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(5*pi/3) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(10.0402576694398) q[1];
rz(pi/2) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(3.15608897149028) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(2*pi) q[8];
cx q[8],q[1];
rz(1.63109410667012) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(5.10318148243128) q[1];
rz(-pi/2) q[1];
h q[1];
rz(-pi/2) q[1];
rz(8.3164741887972) q[1];
rz(-3.08839932966484) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(4.49825482745484) q[8];
rz(-pi/2) q[8];
h q[8];
rz(-pi/2) q[8];
rz(11.4645755905882) q[8];
rz(1.94563165254826) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(6.11431840720457) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(12.2359619585249) q[9];
cx q[3],q[9];
rz(-pi/2) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(4.34020433763047) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(5*pi/2) q[3];
rz(-0.540743674528192) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(5.18778557952427) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(6.93448567436807) q[9];
cx q[3],q[9];
rz(pi/2) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(3.48164460993258) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(2*pi) q[3];
rz(-0.615479708670387) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(5*pi/3) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(10.0402576694398) q[9];
cx q[3],q[9];
rz(-1.45579471196017) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(4.21050082113854) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(6.44766123407778) q[3];
rz(1.69284788183069) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(4.72450770196438) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(7.3200991630266) q[3];
cx q[5],q[3];
rz(-0.523705299288616) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(5.17611038617022) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(6.94219570064781) q[3];
rz(-pi/2) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(4.12369928613164) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(5*pi/2) q[5];
cx q[5],q[3];
rz(-0.615479708670387) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(5*pi/3) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(10.0402576694398) q[3];
rz(-pi/2) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(3.51070753382045) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(3*pi) q[5];
cx q[5],q[3];
rz(2.48617039230287) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(3.71784711262738) q[3];
rz(-pi/2) q[3];
h q[3];
rz(-pi/2) q[3];
rz(8.48382878452177) q[3];
rz(0.96373075272785) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(4.35017118403806) q[5];
rz(-pi/2) q[5];
h q[5];
rz(-pi/2) q[5];
rz(7.32616187068597) q[5];
rz(-2.11724455450658) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(4.67054294897347) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(9.40489218614909) q[9];
rz(0.32380916759648) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(3.78990275083196) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(8.39859108733906) q[9];
cx q[9],q[2];
rz(-0.0630674123017583) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(4.7753313402098) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(7.06659735145785) q[2];
rz(-pi/2) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(3.81993707977298) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(5*pi/2) q[9];
cx q[9],q[2];
rz(-0.615479708670387) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5*pi/3) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(10.0402576694398) q[2];
rz(-pi/2) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(3.30433796388591) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(3*pi) q[9];
cx q[9],q[2];
rz(0.616062947205337) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(5.09872221044305) q[2];
rz(-pi/2) q[2];
h q[2];
rz(-pi/2) q[2];
rz(12.292598721741) q[2];
rz(-2.45655919811552) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(6.16027913899614) q[9];
rz(-pi/2) q[9];
h q[9];
rz(-pi/2) q[9];
rz(8.16791284226335) q[9];