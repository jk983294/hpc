#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

bool checkForErrorA() { return false; }
bool checkForErrorB() { return false; }
bool checkForErrorC() { return false; }
void handleErrorA() {}
void handleErrorB() {}
void handleErrorC() {}
void sendOrder2Exchange() {}
void handleErrors(int64_t errorFlags) {}

void bad() {
    if (checkForErrorA()) {
        handleErrorA();
    } else if (checkForErrorB()) {
        handleErrorB();
    } else if (checkForErrorC()) {
        handleErrorC();
    } else {
        sendOrder2Exchange();
    }
}

/**
 * less branch gives better instruction cache and data cache
 */
void good() {
    int64_t errorFlags = 0;

    // calcuale error flag here

    if (!errorFlags) {
        sendOrder2Exchange();
    } else {
        handleErrors(errorFlags);
    }
}

int main() {
    cout << "hello" << endl;
    return 0;
}
