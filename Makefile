TIMINGDIR = /home/fas/cpsc424/ahs3/utils/timing
CC = icc
MPICC = mpicc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99 -I$(TIMINGDIR)

all: blocking serial nonblocking loadbalancing

loadbalancing: loadbalancing.o matmul.o $(TIMINGDIR)/timing.o
	$(MPICC) -o $@ $(CFLAGS) $^

nonblocking: nonblocking.o matmul.o $(TIMINGDIR)/timing.o
	$(MPICC) -o $@ $(CFLAGS) $^

blocking: blocking.o matmul.o $(TIMINGDIR)/timing.o
	$(MPICC) -o $@ $(CFLAGS) $^

serial:	serial.o matmul.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f serial *.o
