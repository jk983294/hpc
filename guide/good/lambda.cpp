#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

struct Msg {
    double price;
};

Msg PrepareMessage() { return Msg(); }
void send(Msg& msg) {}

template <typename T>
void SendMessage(T&& lambda) {
    Msg msg = PrepareMessage();
    lambda(msg);
    send(msg);
}

/**
 * if you know at compile time which function is to be executed, then prefer lambdas
 * lambda tends to be inlined
 */
int main() {
    double x = 1.2;
    SendMessage([&](auto& msg) { msg.price = x; });
    return 0;
}
