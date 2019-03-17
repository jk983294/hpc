#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

/**
 * ((always_inline)) and ((noinline)) can make your code faster and it can also slow down
 * so do measurement before decision be made
 */

bool CheckMarket() { return false; }
__attribute__((noinline)) void ComplexLoggingFunction() {}
void SendOrder() {}

int main() {
    bool notGoingToSendOrder = CheckMarket();
    if (notGoingToSendOrder) {
        /**
         * here we choose to not inline
         * if inlined, it will pollute the code cache since ComplexLoggingFunction code will be dragged in
         * usually this branch will be ignored
         */
        ComplexLoggingFunction();
    } else {
        SendOrder();
    }
    return 0;
}
