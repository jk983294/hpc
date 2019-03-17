#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

struct OrderSenderA {
    void SendOrder() {
        // ...
    }
};

struct OrderSenderB {
    void SendOrder() {
        // ...
    }
};

struct IOrderManager {
    virtual void MainLoop() = 0;
};

template <typename T>
struct OrderManager : public IOrderManager {
    void MainLoop() final {
        // ...
        mOrderSender.SendOrder();
    }
    T mOrderSender;
};

std::unique_ptr<IOrderManager> Factory(bool useA) {
    if (useA)
        return std::make_unique<OrderManager<OrderSenderA>>();
    else
        return std::make_unique<OrderManager<OrderSenderB>>();
}

int main() {
    bool config = true;
    auto manager = Factory(config);
    manager->MainLoop();
    return 0;
}
