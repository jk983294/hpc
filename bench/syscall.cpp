#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utils/Timer.h>

int main(void) {
    const int iterations = 10000000;
    const uint64_t startNs = ztool::ntime();
    for (int i = 0; i < iterations; i++) {
        if (syscall(SYS_gettid) <= 1) {
            exit(2);
        }
    }
    const uint64_t delta = ztool::ntime() - startNs;
    printf("%i system calls in in %zu ns (%.1f ns/iterations)\n", iterations, delta, (delta / (float)iterations));
    return 0;
}
