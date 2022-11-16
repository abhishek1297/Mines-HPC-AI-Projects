CC = gcc
LIBS = -lm -lpthread -mavx2
#-ffast-math -ftree-vectorize -mveclibabi=svml -mstackrealign 

all:
	${CC} -o 1to4.out ex1to4.c ${LIBS}
	${CC} -o 5.out ex5.c ${LIBS}
clean:
	rm -rf *.out