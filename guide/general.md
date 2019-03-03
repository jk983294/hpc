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
