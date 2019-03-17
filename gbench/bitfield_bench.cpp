#include <benchmark/benchmark.h>
#include <cstring>
#include <memory>

using namespace std;

//------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
//------------------------------------------------------------------
// bench_copy_field              28.6 ns         28.6 ns     24163623
// bench_copy_array              4.39 ns         4.39 ns    159589627
// bench_copy_whole_memcpy       4.39 ns         4.39 ns    159584429

struct Order {
    struct pack_ {
        int key;
        int SnapShotIdx;
        double LimitPrice;
        double TradePrice;
        char StrategyID[24];
        int TradeVolume;
        char InstrumentID[11];
        unsigned int Direction : 2;
        unsigned int OffsetFlag : 2;
        unsigned int OrderStatus : 4;
        unsigned int ExchangeIdx : 4;
        unsigned int __reserved0__ : 28;
        unsigned int SubOEIdx;
        unsigned int SubAccIdx : 8;
        unsigned int PriceType : 2;
        unsigned int __reserved1__ : 22;
        unsigned int __reserved2__;
        int VolumeTraded;
    } pack __attribute__((packed));
    long InsertTime;
    int VolumeTotal;
    char InvestorID[24];
    char OrderSysID[40];
} __attribute__((packed));

struct Recv {
    struct pack_ {
        int key;
        int SnapShotIdx;
        double LimitPrice;
        double TradePrice;
        char StrategyID[24];
        int TradeVolume;
        char SecurityID[11];
        unsigned int Direction : 2;
        unsigned int OffsetFlag : 2;
        unsigned int OrderStatus : 4;
        unsigned int ExchangeIdx : 4;
        unsigned int __reserved0__ : 20;
        unsigned int FieldType : 8;
        unsigned int __reserved1__;
        long UpdateTime;
        int ErrorID;
    } pack __attribute__((packed));
    union {
        int VolumeTraded;
        int VolumeTotal;
    };
} __attribute__((packed));

Order create_order() {
    Order oe;
    memset(&oe, 0, sizeof(oe));
    oe.pack.key = 1;
    oe.pack.SnapShotIdx = 2;
    oe.pack.LimitPrice = 3;
    oe.pack.TradePrice = 4;
    strcpy(oe.pack.StrategyID, "test opt sid");
    oe.pack.TradeVolume = 5;
    strcpy(oe.pack.InstrumentID, "600000.SSE");
    oe.pack.Direction = 1;
    oe.pack.OffsetFlag = 1;
    oe.pack.OrderStatus = 2;
    oe.pack.ExchangeIdx = 3;
    return oe;
}

Recv* FormatRecvFromOrderByField(Order* order, Recv* recv) {
    memset(recv, 0, sizeof(Recv));
    recv->pack.key = order->pack.key;
    recv->pack.SnapShotIdx = order->pack.SnapShotIdx;
    recv->pack.LimitPrice = order->pack.LimitPrice;
    strncpy(recv->pack.StrategyID, order->pack.StrategyID, sizeof(recv->pack.StrategyID));
    strncpy(recv->pack.SecurityID, order->pack.InstrumentID, sizeof(recv->pack.SecurityID));
    recv->pack.Direction = order->pack.Direction;
    recv->pack.OffsetFlag = order->pack.OffsetFlag;
    recv->pack.OrderStatus = order->pack.OrderStatus;
    recv->VolumeTraded = order->pack.VolumeTraded;
    recv->pack.ExchangeIdx = order->pack.ExchangeIdx;
    return recv;
}

Recv* FormatRecvFromOrderByArray(Order* order, Recv* recv) {
    int64_t* addr1 = reinterpret_cast<int64_t*>(order);
    int64_t* addr2 = reinterpret_cast<int64_t*>(recv);
    for (int i = 0; i < 9; ++i) {
        addr2[i] = addr1[i];
    }
    recv->VolumeTraded = order->pack.VolumeTraded;
    return recv;
}

Recv* FormatRecvFromOrderByMemcpy(Order* order, Recv* recv) {
    memcpy(recv, order, sizeof(int64_t) * 9);
    recv->VolumeTraded = order->pack.VolumeTraded;
    return recv;
}

void bench_copy_field(benchmark::State& state) {
    Order order = create_order();
    Recv recv;
    memset(&recv, 0, sizeof(recv));
    for (auto _ : state) {
        benchmark::DoNotOptimize(FormatRecvFromOrderByField(&order, &recv));
    }
}

BENCHMARK(bench_copy_field);

void bench_copy_array(benchmark::State& state) {
    Order order = create_order();
    Recv recv;
    memset(&recv, 0, sizeof(recv));
    for (auto _ : state) {
        benchmark::DoNotOptimize(FormatRecvFromOrderByArray(&order, &recv));
    }
}

BENCHMARK(bench_copy_array);

void bench_copy_whole_memcpy(benchmark::State& state) {
    Order order = create_order();
    Recv recv;
    memset(&recv, 0, sizeof(recv));
    for (auto _ : state) {
        benchmark::DoNotOptimize(FormatRecvFromOrderByMemcpy(&order, &recv));
    }
}

BENCHMARK(bench_copy_whole_memcpy);

BENCHMARK_MAIN();
