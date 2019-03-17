#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

enum class Side { Buy, Sell };

float CalcPrice(Side side, float v, float credit) { return side == Side::Buy ? v - credit : v + credit; }
void SendOrder(Side side, float p) {}
void checkRiskLimits(Side side, float orderPrice) {}

void RunStrategy(Side side) {
    float fairValue = 42, credit = 42;
    const float orderPrice = CalcPrice(side, fairValue, credit);
    checkRiskLimits(side, orderPrice);
    SendOrder(side, orderPrice);
}

/**
 * refactor to template
 * apply it for hot path only
 */
template <Side T>
struct Strategy {
    float fairValue = 42;
    float credit = 42;

    void RunStrategy() {
        const float orderPrice = CalcPrice(fairValue, credit);
        //        checkRiskLimits(orderPrice);  // do the same partial specilization
        //        SendOrder(orderPrice);
    }

    float CalcPrice(float v, float credit);
};

template <>
float Strategy<Side::Buy>::CalcPrice(float v, float credit) {
    return v - credit;
}

template <>
float Strategy<Side::Sell>::CalcPrice(float v, float credit) {
    return v + credit;
}

int main() {
    Strategy<Side::Buy> strategy;
    return 0;
}
