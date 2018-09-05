#include <linux/futex.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/Timer.h>

int main(void) {
    const int iterations = 500000;
    const int shm_id = shmget(IPC_PRIVATE, sizeof(int), IPC_CREAT | 0666);
    const pid_t other = fork();
    int* futex = (int*)shmat(shm_id, NULL, 0);
    *futex = 0xA;
    if (other == 0) {  // child
        for (int i = 0; i < iterations; i++) {
            sched_yield();
            while (syscall(SYS_futex, futex, FUTEX_WAIT, 0xA, NULL, NULL, 42)) {
                sched_yield();
            }
            *futex = 0xB;
            while (!syscall(SYS_futex, futex, FUTEX_WAKE, 1, NULL, NULL, 42)) {
                sched_yield();
            }
        }
        return 0;
    }

    const uint64_t start_ns = flux::ntime();
    for (int i = 0; i < iterations; i++) {
        *futex = 0xA;
        while (!syscall(SYS_futex, futex, FUTEX_WAKE, 1, NULL, NULL, 42)) {
            sched_yield();
        }
        sched_yield();
        while (syscall(SYS_futex, futex, FUTEX_WAIT, 0xB, NULL, NULL, 42)) {
            sched_yield();
        }
    }
    const uint64_t delta = flux::ntime() - start_ns;

    const int nSwitches = iterations * 4;
    printf("%i process context switches in %zu ns ( %.1f ns/ctxsw )\n", nSwitches, delta, (delta / (float)nSwitches));
    wait(futex);
    return 0;
}
