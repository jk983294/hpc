## Efficiency with Algorithms, Performance with Data Structures

### efficiency
an efficient program is one which does the minimum amount of work to accomplish a given task.

algorithm reduce the work amount so that it makes more efficient.

improving algorithmic efficiency requires finding a different way of solving the program.

### performance
how fast program finish the work. the amount of work is the same, but how fast.

lighting up all of transistor.

### data structure abd algorithms

they are tightly coupled

worse is better: bubble sort for small data-set and cuckoo hashing is good because of cache miss

### bad

do not use std::list/std::map/std::unordered_map, in most cases every step is a cache miss, so favor std::vector

### good hash map design

no buckets, use open address into table

table stored as contiguous range of memory

use local probing on collisions to find an open slot in the same cache line (usually)

keep both key and values small

### good
for large object, use identity field for index or pointer in array

pack bits (because computation is cheap there, cpu usually wait for data (around 50%), so pack the data to small data structure is good )

use bitfields everywhere

## do not do more work than you have to
calculate values once - at initilization time

if it looks simpler, it is probably faster

avoid object copying

avoid automatic conversion: 1) do not pass smart pointers (pass its underlying pointer) 2) make conversion operation explicit

avoid std::endl, use '\n' instead so that you gain 9 times faster

## return rule
avoid std::move in your return - it will prevent RVO

return type must be the same as the type you are returning

return a local variable or by value parameter

multiple return statements will often prevent RVO

conditional expressions in return statement will often prevent auto-move

## smaller code is faster
do not repeat yourself in templates, move those to non templated base class

avoid std::shared_ptr

prefer return std::unique_ptr<> from factory

prefer bare function call, no std::function, no std::bind, use lambda instead

## avoid non-local data
1. static data
2. need some kind of mutex protection
3. be in a container with on-trival lookup costs (like std::map<>)

## compile
use final can help compiler optimize virtual function calls

constexpr do compile calculation

## machine
turn off hyper-threading, because two thread share the same cache, it will slow down

## memory allocation
use a pool of pre-allocated objects

reuse objects instead of de-allocating

if you must delete large objects, consider doing this from another thread

## exception in c++
exception are zero cost if they don't throw

do not use exception for branch control, cost at least 1.5us

## multi-threading
keep shared data to an absolute minimum, multiple threads write to the same cache line will get expensive

consider passing copies of data rather than sharing data
* e.g. a single writer, single reader lock free queue

if you have to share data, consider not using synchronization
* e.g. maybe you can live with out of sequence updates

## keep the cache hot
the full hot path is only exercised every infrequently

your cache has most likely been trampled by non hot path data and instruction

simple solution:
* run a very frequent dummy path through your entire system, keeping both your data cache and instruction cache primed
* this solution also trains the hardware branch predictor correctly

## do not share L3
disable all but 1 core (or lock the cache)

if you do have multiple cores enabled, choose your neighbours carefully:
* noisy neighbours should probably be moved to a different physical CPU

## no placement new
it will perform a null pointer check

new version gcc compiler has fixed this, now pass in null to placement new is UB

## small string optimization support
if you use gcc >= 5.1 and an ABI compatible linux distribution such as Redhat/Centos/Ubuntu/Fedora, then you are probably still using old std::string implementation

## std::pow can be slow
glibc 2.21 fix it

double base = 1.000000000001, exp1 = 1.4, exp2 = 1.5;
std::pow(base, exp1);

## optimize path
预取，hugepage，细节trick算法，AVX/BMI/FMA指令优化