#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./pipe_throughput <message-size> <message-count>
 *
 * message size: 128 message count: 1024000 avg throughput: 841587.635 msg/s 821.863 Mb/s
 * message size: 256 message count: 1024000 avg throughput: 844525.596 msg/s 1649.464 Mb/s
 * message size: 512 message count: 1024000 avg throughput: 840515.453 msg/s 3283.263 Mb/s
 * message size: 1024 message count: 1024000 avg throughput: 549204.852 msg/s 4290.663 Mb/s
 * message size: 2048 message count: 1024000 avg throughput: 765609.392 msg/s 11962.647 Mb/s
 * message size: 16384 message count: 1024000 avg throughput: 183537.676 msg/s 22942.210 Mb/s
 * message size: 131072 message count: 1024000 avg throughput: 17136.055 msg/s 17136.055 Mb/s
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char *buf = new char[size];

    int fds[2];
    if (pipe(fds) == -1) {
        perror("pipe");
        return 1;
    }

    if (!fork()) {
        // child read content from parent
        for (int i = 0; i < count; i++) {
            if (read(fds[0], buf, size) != size) {
                perror("read");
                return 1;
            }
        }
    } else {
        // parent send data to child

        uint64_t startNs = ztool::ntime();

        for (int i = 0; i < count; i++) {
            if (write(fds[1], buf, size) != size) {
                perror("write");
                return 1;
            }
        }

        const uint64_t delta = ztool::ntime() - startNs;
        double deltaSecond = ((double)delta) / 1e9;

        cout << std::fixed << setprecision(3) << "message size: " << size << " message count: " << count
             << " avg throughput: " << (count / deltaSecond) << " msg/s "
             << (((count / deltaSecond) * size * 8) / (1024 * 1024)) << " Mb/s" << endl;
    }
    return 0;
}
