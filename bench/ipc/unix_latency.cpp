#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./unix_latency <message-size> <round-trip-count>
 *
 * message size: 128 round trip count: 1024000 avg latency: 2796.7 ns
 * message size: 256 round trip count: 1024000 avg latency: 2757.3 ns
 * message size: 512 round trip count: 1024000 avg latency: 2761.7 ns
 * message size: 1024 round trip count: 1024000 avg latency: 2781.5 ns
 * message size: 2048 round trip count: 1024000 avg latency: 2972.7 ns
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    int sv[2]; /* the pair of socket descriptors */

    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char *buf = new char[size];

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1) {
        perror("socketpair");
        return 1;
    }

    if (!fork()) {  // child
        for (int i = 0; i < count; i++) {
            if (read(sv[1], buf, size) != size) {
                perror("read");
                return 1;
            }

            if (write(sv[1], buf, size) != size) {
                perror("write");
                return 1;
            }
        }
    } else {  // parent

        sleep(1);

        uint64_t startNs = flux::ntime();

        for (int i = 0; i < count; i++) {
            if (write(sv[0], buf, size) != size) {
                perror("write");
                return 1;
            }

            if (read(sv[0], buf, size) != size) {
                perror("read");
                return 1;
            }
        }

        const uint64_t delta = flux::ntime() - startNs;

        printf("message size: %d round trip count: %d avg latency: %.1f ns\n", size, count, delta / ((float)count * 2));
    }

    return 0;
}
