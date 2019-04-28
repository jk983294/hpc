## smartness
* pre-calculate for all possibility and cache them, when something happens, just fire
* move decision from runtime to compile time
* heuristics are acceptable (no need for high precision)

## cache
* general L3 is shared across all cores
* pick your neighbours wisely
* disable HT, Hyper threading share L1
* you want all your data to be in cache
* cache warm, keep running through hot path, keep touching memory

## measure latency
* software timestamp
* hardware timestamp from NIC
* macro benchmark

## implementation
* simpler code, faster it is likely to be
* know your hardware
* bypass the kernel, arm for 100% user-space code, including network IO

## specifics
* read the resultant assembly
* cache warming helps

## code tricks
* data member layout, padding and alignment
* false sharing
* cache locality
* compile time dispatch (std::sort faster than qsort)
* constexpr
* variadic template (maciekgajewski/Fast-Log)
* loop un-rolling, don't do it manually, compiler can take care of it
* expression short-circuiting, do inexpensive check before expensive check
* signed vs unsigned comparison, use int in for loop, because size_t will check overflow
* don't mix float/double conversion, -ffast-math
* reduce if else number
* static link tend to help
* debug symbols have little impact, those sits in non-cached memory until a core dump occurs
* machine-specific compiler flags -march -mtune
* kernel bypass, user space network like OpenOnload
* cache warm
* process control, thread affinity, memory affinity