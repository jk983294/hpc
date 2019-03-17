#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

struct Market {
    int32_t id;
    char shortName[4];
    int16_t quantityMultiplier;
    // more fields
};

struct Instrument {
    float price;
    /**
     * !!! good, data de-normalized, put all data in the same cache line
     * better than trampling your cache to "save memory"
     */
    int16_t quantityMultiplier;  // extract from Market
    // more fields
    int32_t marketId;
};

struct Message {
    float price;
    int qty;
};

struct Markets {
    std::unordered_map<int32_t, Market> markets;

    Market& FindMarket(int32_t idx) { return markets[idx]; }
};

void bad() {
    int qty = 42;
    Message orderMessage;
    Instrument instrument;
    orderMessage.price = instrument.price;
    Markets markets;
    Market& market = markets.FindMarket(instrument.marketId);
    orderMessage.qty = market.quantityMultiplier * qty;
}

void good() {
    int qty = 42;
    Message orderMessage;
    Instrument instrument;
    orderMessage.price = instrument.price;
    orderMessage.qty = instrument.quantityMultiplier * qty;  // only need to read instrument
}

int main() { return 0; }
