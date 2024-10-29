CC = g++
OUT = lab_3.out
CFLAGS = `pkg-config --cflags --libs opencv4`

DEPS = ${wildcard *.h}
SRCS = ${wildcard *.cpp}

all: default


default:
	${CC} ${SRCS} ${DEPS} -o ${OUT} ${CFLAGS}

clean:
	rm -f ${OUT}


