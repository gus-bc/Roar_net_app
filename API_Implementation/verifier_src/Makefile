PROG = ./coloring-verifier
#CFLAGS = -Wall -static -ansi -pedantic
#LDFLAGS = -lm  

OBJ=bin2asc.o utilities.o checker.o

$(PROG): $(OBJ)
	$(CC) $(CFLAGS) -o $(PROG) $(LDFLAGS) $(OBJ) main.c 

clean:
	rm -f *~ *.o $(PROG)


test:
	${PROG} -i ../examples/marco10.col -s ../examples/marco10.sol -p 1
