#include <arpa/inet.h>
#include <benchmark/benchmark.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

/**
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
MyFixture/write_syscall                1338 ns         1338 ns       436361
MyFixture/prepare_data_to_buffer       22.3 ns         22.3 ns     31446529
MyFixture/write_twice                  2786 ns         2786 ns       235945
MyFixture/write_once_with_buffer       930 ns          930 ns        668407

result interpretation:
compare to socket write syscall, memcpy cost very little
try to compact all data you want to send using memcpy, then launch one write syscall
*/

constexpr size_t size{1024};

struct Header {
    int data[5];
};
struct Body {
    int data[50];
};

struct TcpSender {
    int round{-1}, sockfd;
    string server_port{"28023"};
    char buf[size];
    int count{0};
    Header header;
    Body body;

    bool init() {
        header.data[0] = 0;
        body.data[0] = 0;
        struct addrinfo* res;
        struct addrinfo hints;
        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_UNSPEC;  // use IPv4 or IPv6, whichever
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;  // fill in my IP for me

        int ret = getaddrinfo("0.0.0.0", server_port.c_str(), &hints, &res);
        if (ret != 0) {
            fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
            return false;
        }

        if ((sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol)) == -1) {
            perror("socket");
            return false;
        }

        if (connect(sockfd, res->ai_addr, res->ai_addrlen) == -1) {
            perror("connect");
            return false;
        }
        int on = 1;
        if (setsockopt(sockfd, SOL_TCP, TCP_NODELAY, &on, sizeof(on)) == -1) {
            perror("setsockopt TCP_NODELAY");
            return false;
        }
        cout << "connect to 0:" << server_port << endl;
        return true;
    }

    ssize_t write_socket(char const* _buf, size_t _send_len) {
        std::size_t sentCount = 0;
        std::size_t unsentCount = _send_len;
        while (unsentCount > 0) {
            ssize_t nsnt = send(sockfd, _buf + sentCount, unsentCount, MSG_NOSIGNAL);
            if (nsnt > 0) {
                sentCount += nsnt;
                unsentCount -= nsnt;
            } else if (nsnt < 0) {
                if ((EINTR != errno) && (EAGAIN != errno)) {
                    return -1;
                }
            }
        }
        return sentCount;
    }

    int write_twice() {
        prepare_data();
        int ret = write_socket(reinterpret_cast<char*>(&header), sizeof(header));
        ret += write_socket(reinterpret_cast<char*>(&body), sizeof(body));
        return ret;
    }

    int write_once_with_buffer() {
        prepare_data_to_buffer();
        return write_socket(buf, sizeof(header) + sizeof(body));
    }

    void prepare_data() {
        ++count;
        header.data[0] = count;
        body.data[0] = count;
    }

    int prepare_data_to_buffer() {
        prepare_data();
        memcpy(buf, &header, sizeof(header));
        memcpy(buf + sizeof(header), &body, sizeof(body));
        return body.data[0];
    }
};

class MyFixture : public benchmark::Fixture {
public:
    MyFixture() { sender.init(); }

    ~MyFixture() {}

    void SetUp(const ::benchmark::State& state) { sender.count = 0; }

    void TearDown(const ::benchmark::State& state) {}

    TcpSender sender;
};

// BENCHMARK_F(MyFixture, write_syscall)(benchmark::State& st) {
//    for (auto _ : st) {
//        benchmark::DoNotOptimize(sender.write_socket(sender.buf, 1));
//    }
//}
//
// BENCHMARK_F(MyFixture, prepare_data_to_buffer)(benchmark::State& st) {
//    for (auto _ : st) {
//        benchmark::DoNotOptimize(sender.prepare_data_to_buffer());
//    }
//}
//
// BENCHMARK_F(MyFixture, write_twice)(benchmark::State& st) {
//    for (auto _ : st) {
//        benchmark::DoNotOptimize(sender.write_twice());
//    }
//}

BENCHMARK_F(MyFixture, write_once_with_buffer)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(sender.write_once_with_buffer());
    }
}

BENCHMARK_MAIN();
