#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./tcp_local_latency <host> <port> <message-size> <round-trip-count>
 * remote work as a client, connect to server, then send data to server, then read from server, then echo back
 *
 * message size: 128 round trip count: 1024000 avg latency: 20995.3 ns
 * message size: 256 round trip count: 1024000 avg latency: 21431.2 ns
 * message size: 512 round trip count: 1024000 avg latency: 24836.8 ns
 * message size: 1024 round trip count: 1024000 avg latency: 27919.2 ns
 * message size: 2048 round trip count: 1024000 avg latency: 45047.0 ns
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    int sockfd;

    if (argc == 5) {
        size = atoi(argv[3]);
        count = atol(argv[4]);
    } else if (argc == 4) {
        size = atoi(argv[3]);
    } else {
        cout << "arg count not correct!" << endl
             << "\tusage: ./tcp_remote_latency <server host> <port> <message-size> <round-trip-count>\n"
             << "\texample: ./tcp_remote_latency 10.18.0.191 8023 1024 1024\n";
        return -1;
    }

    char *buf = new char[size];

    struct sockaddr_in serverAddr;
    int portNo = atoi(argv[2]);
    if (!portNo) {
        servent *s = getservbyname(argv[2], "tcp");
        if (!s) {
            perror("getservbyname() failed");
            return 1;
        }
        portNo = ntohs(static_cast<uint16_t>(s->s_port));
    }

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        return 1;
    }

    struct hostent *server = gethostbyname(argv[1]);  // DNS resolve to server info
    if (!server) {
        perror("gethostbyname failed, no such host");
        return 1;
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    memcpy(&serverAddr.sin_addr.s_addr, server->h_addr_list[0], server->h_length);
    serverAddr.sin_port = htons(static_cast<uint16_t>(portNo));

    if (connect(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
        perror("connect");
        return 1;
    }

    uint64_t startNs = flux::ntime();

    for (int i = 0; i < count; i++) {
        if (write(sockfd, buf, size) != size) {
            perror("write");
            return 1;
        }

        for (int sofar = 0; sofar < size;) {
            int len = read(sockfd, buf, size - sofar);
            if (len == -1) {
                perror("read");
                return 1;
            }
            sofar += len;
        }
    }

    const uint64_t delta = flux::ntime() - startNs;

    printf("message size: %d round trip count: %d avg latency: %.1f ns\n", size, count, delta / ((float)count * 2));
    return 0;
}
