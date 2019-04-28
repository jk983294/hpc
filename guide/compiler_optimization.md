## inline 
* lambda get more change of inline than free function, sort with lambda used to be faster than free function
* flag --max-inline-instns-auto=100 --early-inlining-instns=200
* be care of destructor inlined, control by __always_inline__, __noinline__
* __noinline__ not important code like error handling

## const propagation
* it helps optimization

## pass by reference
* pass primitive like bool by reference is bad

## branch
* use template with bool to reduce branch, the actual used code will get into ICache
* branch prediction

## loop
* -ftree-loop-distribute-patterns   make one loop to several loops, each has better locality
* -funroll-loops                    space/time trade-off

## compiler options
* -fstrict-aliasing
* gcc -O2 -Q --help=optimizers      check what optimizer enabled
* -march=native -march=x86-64       choose correct machine architecture
