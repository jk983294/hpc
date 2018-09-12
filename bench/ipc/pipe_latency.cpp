#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./pipe_latency <message-size> <round-trip-count>
 *
 * message size: 128 round trip count: 1024000 avg latency: 4268.8 ns
 * message size: 256 round trip count: 1024000 avg latency: 4394.9 ns
 * message size: 512 round trip count: 1024000 avg latency: 4557.7 ns
 * message size: 1024 round trip count: 1024000 avg latency: 4690.9 ns
 * message size: 2048 round trip count: 1024000 avg latency: 4868.6 ns
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024;

    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char *buf = new char[size];

    int fdSet1[2];
    int fdSet2[2];

    if (pipe(fdSet1) == -1) {
        perror("pipe");
        return 1;
    }
    if (pipe(fdSet2) == -1) {
        perror("pipe");
        return 1;
    }

    if (!fork()) {
        /**
         * child first wait for parent write to fdSet1[1], then read content from fdSet1[0], then write back to
         * fdSet2[1]
         */
        for (int i = 0; i < count; i++) {
            if (read(fdSet1[0], buf, size) != size) {
                perror("read");
                return 1;
            }

            if (write(fdSet2[1], buf, size) != size) {
                perror("write");
                return 1;
            }
        }
    } else {  // parent
        uint64_t startNs = flux::ntime();

        /**
         * parent first write to fdSet1[1], then wait child to echo back via fdSet2[0]
         */
        for (int i = 0; i < count; i++) {
            if (write(fdSet1[1], buf, size) != size) {
                perror("write");
                return 1;
            }

            if (read(fdSet2[0], buf, size) != size) {
                perror("read");
                return 1;
            }
        }

        const uint64_t delta = flux::ntime() - startNs;

        printf("message size: %d round trip count: %d avg latency: %.1f ns\n", size, count, delta / ((float)count * 2));
    }

    return 0;
}
