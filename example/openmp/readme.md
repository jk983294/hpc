## SPMD pattern
single program multiply data

you use the same code to loop over different set of data, then sum the result back to get the final result

use thread id to select a set of tasks to manage shared data

## loop pattern
for this pattern to work, each iteration should not have dependency

## divide and conquer pattern
divide problem into sub-problems and recombine solutions of sub-problems into a global solution

usually use single construct to pick one thread to create tasks and the rest threads to pick up task from queue

### schedule
how loop iterations are mapped onto threads
* schedule(static[, chunk]),     split into chunks to feed thread, chunk is optional, if not provided, openmp will figure out
* schedule(dynamic[, chunk]),    it split into chunks and queue them in task queue, thread will grab from queue if it is free
* schedule(guided[, chunk]),     firstly split big chunk, then chunk size go down along with time
* schedule(runtime),             schedule with env variable or other configs at runtime
* schedule(auto),                schedule as whatever compiler like, no hint give to compiler

### reduction
reduction(op : list)

a local copy of each list variable is made and initialized depending on the "op"

then updates occur on the local copy

at last, local copies are reduced into a single value and combined with the original global value

how local copies reduced into single value, it also depend on "op"

op list to initial value:
* \+ -> 0
* \* -> 1
* \- -> 0
* min -> largest positive number
* max -> most negative number
* & -> ~0
* | -> 0
* ^ -> 0
* && -> 1
* || -> 0

## environment variable
* OMP_NUM_THREADS
* OMP_STACKSIZE
* OMP_WAIT_POLICY, ACTIVE|PASSIVE, spin lock|sleep
* OMP_PROC_BIND, TRUE|FALSE

## data environment
by default, local variable within parallel block is private, the rest are shared

change storage attribute:
* SHARED
* PRIVATE, create a private variable (un-initialized) to shadow the global one in each thread
* FIRSTPRIVATE, create a private variable (initialized to global variable) to shadow the global one in each thread
* LASTPRIVATE, the last iteration's value will be copied to global variable
* DEFAULT(SHARED), this is default, all variable declared outside block is shared
* DEFAULT(NONE), force you declare all attribute
* reduction(op : list)

## memory model
flush list is guaranteed to consist at flush point
flush = fence
