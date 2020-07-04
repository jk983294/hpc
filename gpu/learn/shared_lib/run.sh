rm -f *.o *.so *.a main a.out
#  option -x cu makes allows nvcc to handle cpp like cu-files

# rdc must follow dlink for separate compilation
nvcc -arch=sm_61 -rdc=true -c a.cu
nvcc -arch=sm_61 -dlink -o a_link.o a.o -lcudadevrt -lcudart
g++ a.o a_link.o main.cpp -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

# non separate compilation
nvcc -arch=sm_61 -c a.cu
g++ a.o main.cpp -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

# static lib
nvcc -arch=sm_61 -Xcompiler -fPIC -rdc=true --lib a.cu -o libtest_static.a -I.
nvcc -o main main.cpp -L. -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt -ltest_static

# demo kernel.cu that is calling device code in another compilation unit myclass.cpp
# nvcc -Xcompiler -fPIC -x cu -c -dc -o myclass.o myclass.cpp
# nvcc -Xcompiler -fPIC -rdc=true --lib myclass.o kernel.cu -o libhelpme.a -I.
# nvcc -o program main.cc -I. -L. -lhelpme

# mix static and shared-object libraries when linking
# gcc foo.c -Wl,-Bstatic -lbar -lbaz -lqux -Wl,-Bdynamic -lcorge -o foo

# dynamic lib
nvcc -arch=sm_61 -Xcompiler -fPIC -I. -L. -rdc=true -x cu -c a.cu -o a.o
nvcc -arch=sm_61 -Xcompiler -fPIC -o libtest_dynamic.so --shared a.o
nvcc -o main main.cpp -Xlinker=-rpath,. -L. -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt -ltest_dynamic
