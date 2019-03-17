#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./fifo_latency <message-size> <round-trip-count>
 *
 * message size: 128 round trip count: 1024 avg latency: 2511.7 ns
 * message size: 256 round trip count: 1024 avg latency: 2088.2 ns
 * message size: 512 round trip count: 1024 avg latency: 2143.3 ns
 * message size: 1024 round trip count: 1024 avg latency: 2456.5 ns
 * message size: 2048 round trip count: 1024 avg latency: 2235.8 ns
 */

int main(int argc, char* argv[]) {
    int size = 1024;
    int count = 1024;

    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char* buf = new char[size];

    const char* fifoFile1 = "./fifo_ipc1";
    const char* fifoFile2 = "./fifo_ipc2";
    unlink(fifoFile1);
    unlink(fifoFile2);
    if (mkfifo(fifoFile1, 0644) == -1) {
        perror("mkfifo1");
        return 1;
    }
    if (mkfifo(fifoFile2, 0644) == -1) {
        perror("mkfifo2");
        return 1;
    }

    int fd1 = open(fifoFile1, O_RDWR);
    if (fd1 == -1) {
        perror("open1");
        return 1;
    }
    int fd2 = open(fifoFile1, O_RDWR);
    if (fd2 == -1) {
        perror("open2");
        return 1;
    }

    if (!fork()) {
        /**
         * child first wait for parent write to fdSet1[1], then read content from fdSet1[0], then write back to
         * fdSet2[1]
         */
        for (int i = 0; i < count; i++) {
            if (read(fd1, buf, size) != size) {
                perror("read");
                return 1;
            }

            if (write(fd2, buf, size) != size) {
                perror("write");
                return 1;
            }
        }
    } else {  // parent
        uint64_t startNs = ztool::ntime();

        /**
         * parent first write to fdSet1[1], then wait child to echo back via fdSet2[0]
         */
        for (int i = 0; i < count; i++) {
            if (write(fd1, buf, size) != size) {
                perror("write");
                return 1;
            }

            if (read(fd2, buf, size) != size) {
                perror("read");
                return 1;
            }
        }

        const uint64_t delta = ztool::ntime() - startNs;

        printf("message size: %d round trip count: %d avg latency: %.1f ns\n", size, count, delta / ((float)count * 2));
    }

    return 0;
}
