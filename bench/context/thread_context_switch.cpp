#include <linux/futex.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/Timer.h>

constexpr int iterations = 500000;

static void* thread(void* ftx) {
    int* futex = (int*)ftx;
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
    return NULL;
}

int main(void) {
    const int shm_id = shmget(IPC_PRIVATE, sizeof(int), IPC_CREAT | 0666);
    int* futex = (int*)shmat(shm_id, NULL, 0);
    pthread_t thd;
    if (pthread_create(&thd, NULL, thread, futex)) {
        return 1;
    }
    *futex = 0xA;

    const uint64_t startNs = ztool::ntime();
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
    const uint64_t delta = ztool::ntime() - startNs;

    const int nSwitches = iterations * 4;
    printf("%i  thread context switches in %zu ns (%.1f ns/ctxsw)\n", nSwitches, delta, (delta / (float)nSwitches));
    wait(futex);
    return 0;
}
