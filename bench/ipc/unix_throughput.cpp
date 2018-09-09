#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./unix_throughput <message-size> <message-count>
 *
 * message size: 128 message count: 1024000 avg throughput: 1422826.188 msg/s 1389.479 Mb/s
 * message size: 256 message count: 1024000 avg throughput: 1268191.047 msg/s 2476.936 Mb/s
 * message size: 512 message count: 1024000 avg throughput: 1259551.424 msg/s 4920.123 Mb/s
 * message size: 1024 message count: 1024000 avg throughput: 1209342.654 msg/s 9447.989 Mb/s
 * message size: 2048 message count: 1024000 avg throughput: 970590.119 msg/s 15165.471 Mb/s
 * message size: 16384 message count: 1024000 avg throughput: 395946.947 msg/s 49493.368 Mb/s
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    int fds[2]; /* the pair of socket descriptors */

    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char *buf = new char[size];

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == -1) {
        perror("socketpair");
        return 1;
    }

    if (!fork()) {  // child

        for (int i = 0; i < count; i++) {
            if (read(fds[1], buf, size) != size) {
                perror("read");
                return 1;
            }
        }
    } else {  // parent
        sleep(1);
        uint64_t startNs = flux::ntime();

        for (int i = 0; i < count; i++) {
            if (write(fds[0], buf, size) != size) {
                perror("write");
                return 1;
            }
        }

        const uint64_t delta = flux::ntime() - startNs;
        double deltaSecond = ((double)delta) / 1e9;

        cout << std::fixed << setprecision(3) << "message size: " << size << " message count: " << count
             << " avg throughput: " << (count / deltaSecond) << " msg/s "
             << (((count / deltaSecond) * size * 8) / (1024 * 1024)) << " Mb/s" << endl;
    }

    return 0;
}
