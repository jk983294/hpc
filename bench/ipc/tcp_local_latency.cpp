#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "utils/Timer.h"

using namespace std;

/**
 * usage: ./tcp_local_latency <port> <message-size> <round-trip-count>
 * local work as a server, wait for remote to connect, then read data from remote, then echo back
 */

int main(int argc, char *argv[]) {
    int size = 1024;
    int count = 1024000;

    int yes = 1;
    int sockfd, new_fd;

    if (argc == 4) {
        size = atoi(argv[2]);
        count = atol(argv[3]);
    } else if (argc == 3) {
        size = atoi(argv[2]);
    } else {
        cout << "arg count not correct!" << endl
             << "\tusage: ./tcp_local_latency <port> <message-size> <round-trip-count>" << endl
             << "\texample: ./tcp_local_latency 8023 1024 1024" << endl;
        return -1;
    }

    char *buf = new char[size];

    struct sockaddr_in serverAddr, clientAddr;

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        return 1;
    }

    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
        perror("setsockopt");
        return 1;
    }

    int portNo = atoi(argv[1]);
    if (!portNo) {
        servent *s = getservbyname(argv[1], "tcp");
        if (!s) {
            perror("getservbyname() failed");
            return 1;
        }
        portNo = ntohs(static_cast<uint16_t>(s->s_port));
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(static_cast<uint16_t>(portNo));

    if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
        perror("bind");
        return 1;
    }

    if (listen(sockfd, 1) == -1) {
        perror("listen");
        return 1;
    }

    socklen_t addr_size = sizeof clientAddr;

    if ((new_fd = accept(sockfd, (struct sockaddr *)&clientAddr, &addr_size)) == -1) {
        perror("accept");
        return 1;
    }

    for (int i = 0; i < count; i++) {
        for (int sofar = 0; sofar < size;) {
            int len = read(new_fd, buf, size - sofar);
            if (len == -1) {
                perror("read");
                return 1;
            }
            sofar += len;
        }

        if (write(new_fd, buf, size) != size) {
            perror("write");
            return 1;
        }
    }
    return 0;
}
