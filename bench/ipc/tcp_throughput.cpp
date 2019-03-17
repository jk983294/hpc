#include <netdb.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./tcp_throughput <message-size> <message-count>
 *
 * message size: 128 message count: 1024000 avg throughput: 390043.051 msg/s 380.901 Mb/s
 * message size: 256 message count: 1024000 avg throughput: 393504.951 msg/s 768.564 Mb/s
 * message size: 512 message count: 1024000 avg throughput: 378145.685 msg/s 1477.132 Mb/s
 * message size: 1024 message count: 1024000 avg throughput: 419104.632 msg/s 3274.255 Mb/s
 * message size: 2048 message count: 1024000 avg throughput: 393360.475 msg/s 6146.257 Mb/s
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    int yes = 1;
    int ret;
    struct sockaddr_storage their_addr;
    socklen_t addr_size;
    struct addrinfo hints;
    struct addrinfo *res;
    int sockfd, new_fd;

    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char *buf = new char[size];

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;  // use IPv4 or IPv6, whichever
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;  // fill in my IP for me
    if ((ret = getaddrinfo("127.0.0.1", "8023", &hints, &res)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
        return 1;
    }

    if (!fork()) {
        /**
         * child work as a server, wait for parent to connect, then read data from parent, then echo back
         */
        if ((sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol)) == -1) {
            perror("socket");
            return 1;
        }

        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
            perror("setsockopt");
            return 1;
        }

        if (bind(sockfd, res->ai_addr, res->ai_addrlen) == -1) {
            perror("bind");
            return 1;
        }

        if (listen(sockfd, 1) == -1) {
            perror("listen");
            return 1;
        }

        addr_size = sizeof their_addr;

        if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &addr_size)) == -1) {
            perror("accept");
            return 1;
        }

        for (int sofar = 0; sofar < (count * size);) {
            int len = read(new_fd, buf, size);
            if (len == -1) {
                perror("read");
                return 1;
            }
            sofar += len;
        }
    } else {
        /**
         * parent connect to child, then send data to child, then read from child, then echo back
         */

        sleep(1);

        if ((sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol)) == -1) {
            perror("socket");
            return 1;
        }

        if (connect(sockfd, res->ai_addr, res->ai_addrlen) == -1) {
            perror("connect");
            return 1;
        }

        if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(int)) == -1) {
            perror("setsockopt");
            return 1;
        }

        uint64_t startNs = ztool::ntime();

        for (int i = 0; i < count; i++) {
            if (write(sockfd, buf, size) != size) {
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
