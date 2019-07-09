NVCC=/usr/local/cuda/bin/nvcc -g

exp : src/main.cpp graph.o check.o graph.o check.o coding.o pr.o
	${NVCC} -Xcompiler -fopenmp src/main.cpp obj/graph.o obj/check.o obj/coding.o obj/pr.o -o exp

graph.o check.o graph.o check.o coding.o: | obj

obj:
	mkdir -p obj

graph.o : src/graph.cpp src/graph.h
	g++ -c  src/graph.cpp  -o obj/graph.o -fopenmp

check.o : src/check.cpp src/graph.h
	g++ -c src/check.cpp -o obj/check.o 

coding.o : src/codingIndex.cpp src/graph.h
	g++ -c src/codingIndex.cpp -o obj/coding.o 

pr.o : src/pr.cu src/graph.h src/timer.h
	${NVCC} -c -Xcompiler -fopenmp -o obj/pr.o src/pr.cu