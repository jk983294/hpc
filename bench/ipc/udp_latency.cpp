#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./udp_latency <message-size> <round-trip-count>
 *
 * message size: 128 round trip count: 1024000 avg latency: 4133.1 ns
 * message size: 256 round trip count: 1024000 avg latency: 4132.1 ns
 * message size: 512 round trip count: 1024000 avg latency: 4106.0 ns
 * message size: 1024 round trip count: 1024000 avg latency: 4252.3 ns
 * message size: 2048 round trip count: 1024000 avg latency: 4589.4 ns
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    int yes = 1;
    int ret;
    struct addrinfo hints;
    struct addrinfo *resChild;
    struct addrinfo *resParent;
    int sockfd;
    if (argc == 3) {
        size = atoi(argv[1]);
        count = atol(argv[2]);
    } else if (argc == 2) {
        size = atoi(argv[1]);
    }

    char *buf = new char[size];

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;  // use IPv4 or IPv6, whichever
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;  // fill in my IP for me
    if ((ret = getaddrinfo("127.0.0.1", "8023", &hints, &resParent)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
        return 1;
    }
    if ((ret = getaddrinfo("127.0.0.1", "8024", &hints, &resChild)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
        return 1;
    }

    if (!fork()) {
        /**
         * read data from parent first, then echo back
         */
        if ((sockfd = socket(resChild->ai_family, resChild->ai_socktype, resChild->ai_protocol)) == -1) {
            perror("socket");
            return 1;
        }

        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
            perror("setsockopt");
            return 1;
        }

        if (bind(sockfd, resChild->ai_addr, resChild->ai_addrlen) == -1) {
            perror("bind");
            return 1;
        }

        for (int i = 0; i < count; i++) {
            for (int sofar = 0; sofar < size;) {
                int len = recvfrom(sockfd, buf, size - sofar, 0, resParent->ai_addr, &resParent->ai_addrlen);
                if (len == -1) {
                    perror("recvfrom");
                    return 1;
                }
                sofar += len;
            }

            if (sendto(sockfd, buf, size, 0, resParent->ai_addr, resParent->ai_addrlen) != size) {
                perror("sendto");
                return 1;
            }
        }
    } else {
        /**
         * then send data to child first, then read from child, then echo back
         */
        sleep(1);

        if ((sockfd = socket(resParent->ai_family, resParent->ai_socktype, resParent->ai_protocol)) == -1) {
            perror("socket");
            return 1;
        }

        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
            perror("setsockopt");
            return 1;
        }

        if (bind(sockfd, resParent->ai_addr, resParent->ai_addrlen) == -1) {
            perror("bind");
            return 1;
        }

        uint64_t startNs = ztool::ntime();

        for (int i = 0; i < count; i++) {
            if (sendto(sockfd, buf, size, 0, resChild->ai_addr, resChild->ai_addrlen) != size) {
                perror("sendto");
                return 1;
            }

            for (int sofar = 0; sofar < size;) {
                int len = recvfrom(sockfd, buf, size - sofar, 0, resChild->ai_addr, &resChild->ai_addrlen);
                if (len == -1) {
                    perror("read");
                    return 1;
                }
                sofar += len;
            }
        }

        const uint64_t delta = ztool::ntime() - startNs;

        printf("message size: %d round trip count: %d avg latency: %.1f ns\n", size, count, delta / ((float)count * 2));
    }

    return 0;
}
