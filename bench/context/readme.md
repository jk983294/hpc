Benchmarks the overhead of context switching between 2 threads, by using a shed_yield() method.

If you do taskset -a 1, all threads should be scheduled on the same processor, so you are really doing thread context switch.

Then to be sure that you are really doing it, just do:

```
strace -ff -tt -v taskset -a 1 ./bench_thread_context_switch
```

Now why sched_yield() is enough for testing ? Because, it place the current thread at the end of the ready queue. So the next ready thread will be scheduled.

I also added sched_setscheduler(SCHED_FIFO) to get the best performances.
