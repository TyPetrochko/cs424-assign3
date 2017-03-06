module load Langs/Intel/15 MPI/OpenMPI/1.8.6-intel15;

rm blocking_2.txt blocking_4.txt blocking_8.txt;
rm nonblocking_1.txt nonblocking_2.txt nonblocking_4.txt nonblocking_8.txt;
rm loadbalancing_1.txt loadbalancing_2.txt loadbalancing_4.txt loadbalancing_8.txt;

mpiexec -n 2 blocking > blocking_2.txt &&
mpiexec -n 4 blocking > blocking_4.txt &&
mpiexec -n 8 blocking > blocking_8.txt &&
mpiexec -n 1 nonblocking > nonblocking_1.txt &&
mpiexec -n 2 nonblocking > nonblocking_2.txt &&
mpiexec -n 4 nonblocking > nonblocking_4.txt &&
mpiexec -n 8 nonblocking > nonblocking_8.txt &&
mpiexec -n 1 loadbalancing > loadbalancing_1.txt &&
mpiexec -n 2 loadbalancing > loadbalancing_2.txt &&
mpiexec -n 4 loadbalancing > loadbalancing_4.txt &&
mpiexec -n 8 loadbalancing > loadbalancing_8.txt
