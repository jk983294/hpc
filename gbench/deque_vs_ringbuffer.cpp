#include <benchmark/benchmark.h>
#include <math_random.h>
#include <math_stats_rolling_rb.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <vector>

using namespace std;
using namespace ornate;

/**
 * rb > deque by factor 1.26
 *
 * MyFixture/rb_test/100/65536        482742 ns       482744 ns         1481 items_per_second=33.1439k/s
 * MyFixture/deque_test/100/65536     524672 ns       524687 ns         1244 items_per_second=30.4944k/s
 */

struct rolling_var_deque {
    double total_sum{0}, total_square_sum{0};
    double mean{NAN}, variance{NAN};
    std::deque<double> m_data;
    const int window_size;
    int m_count{0}, m_valid_count{0};

    rolling_var_deque(int size) : window_size{size} {}

    double operator()(double data) {
        m_data.push_back(data);
        ++m_count;

        if (m_count > window_size) {
            double old_value = m_data.front();
            m_data.pop_front();
            if (std::isfinite(old_value)) {
                total_sum -= old_value;
                total_square_sum -= old_value * old_value;
                --m_valid_count;
            }
        }

        if (std::isfinite(data)) {
            total_sum += data;
            total_square_sum += data * data;
            ++m_valid_count;
        }

        if (m_valid_count > 1) {
            mean = total_sum / m_valid_count;
            variance = (total_square_sum - mean * mean * m_valid_count) / (m_valid_count - 1);
        } else {
            mean = NAN;
            variance = NAN;
        }
        return variance;
    }
};

class MyFixture : public benchmark::Fixture {
public:
    MyFixture() {}

    ~MyFixture() {}

    void SetUp(const ::benchmark::State& state) {
        window_size = state.range(0);
        int64_t _count = state.range(1);
        rvrb = new rolling_variance_rb(window_size);
        rvdq = new rolling_var_deque(window_size);
        data = generate_uniform_float(_count, -1.0f, 1.0f);
    }

    void TearDown(const ::benchmark::State& state) {
        delete rvrb;
        delete rvdq;
    }

    template <typename TCalc>
    int64_t _test(TCalc& m) {
        double ret = 0;
        for (auto d : data) {
            ret += m(d);
        }
        return ret;
    }

    rolling_variance_rb* rvrb{nullptr};
    rolling_var_deque* rvdq{nullptr};
    int window_size{5};
    vector<float> data;
};

BENCHMARK_DEFINE_F(MyFixture, rb_test)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(_test(*rvrb));
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, rb_test)
    ->Args({5, 1 << 10})
    ->Args({5, 1 << 12})
    ->Args({5, 1 << 14})
    ->Args({5, 1 << 16})
    ->Args({100, 1 << 10})
    ->Args({100, 1 << 12})
    ->Args({100, 1 << 14})
    ->Args({100, 1 << 16});

BENCHMARK_DEFINE_F(MyFixture, deque_test)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(_test(*rvdq));
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, deque_test)
    ->Args({5, 1 << 10})
    ->Args({5, 1 << 12})
    ->Args({5, 1 << 14})
    ->Args({5, 1 << 16})
    ->Args({100, 1 << 10})
    ->Args({100, 1 << 12})
    ->Args({100, 1 << 14})
    ->Args({100, 1 << 16});

BENCHMARK_MAIN();
