CC = g++
OUT = lab_6.out
CFLAGS = `pkg-config --cflags --libs opencv4` -O0 -pthread
PERF_CPP_DIR = external/perf-cpp
PERF_CPP_LIB = $(PERF_CPP_DIR)/build/libperf.a
PERF_CPP_INCLUDE = -I$(PERF_CPP_DIR)/include
	
DEPS = ${wildcard *.h}
SRCS = ${wildcard *.cpp}
	
all: build_perf_cpp default
	
build_perf_cpp:
	cmake -S $(PERF_CPP_DIR) -B $(PERF_CPP_DIR)/build 
	cmake --build $(PERF_CPP_DIR)/build
	
default:
	${CC} ${SRCS} ${DEPS} -o ${OUT} ${CFLAGS}
	
clean:
	rm -f ${OUT} 
	rm -rf $(PERF_CPP_DIR)/build
	
	
