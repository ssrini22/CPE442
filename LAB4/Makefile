CC = g++
OUT = lab_4.out
CFLAGS = `pkg-config --cflags --libs opencv4` -pthread

DEPS = ${wildcard *.h}
SRCS = ${wildcard *.cpp}

all: default


default:
	${CC} ${SRCS} ${DEPS} -o ${OUT} ${CFLAGS}

clean:
	rm -f ${OUT}



