module load Langs/Intel/15 MPI/OpenMPI/1.8.6-intel15;

./serial > serial.txt &&
mpiexec -n 1 blocking > blocking_1.txt &&
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
mpiexec -n 8 loadbalancing > loadbalancing_8.txt &&
mpiexec -n 7 general > general.txt &&

