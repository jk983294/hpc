#include <benchmark/benchmark.h>
#include <math_random.h>
#include <math_stats_rolling_rb.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <vector>

using namespace ornate;
using namespace std;

/**
 * range > single by factor 1.5, 本质是向量化，按每个tick来更新stats vector, 中间变量内聚到 struct
 *
 * MyFixture/single_test__/100/65536 3566551099 ns   3566666409 ns            1 items_per_second=4.48598/s
 * MyFixture/range_test__/100/65536  2938079940 ns   2938156842 ns            1 items_per_second=5.44559/s
 * MyFixture/range1_test__/100/65536 2213254380 ns   2213192483 ns            1 items_per_second=7.22938/s
 */

int stock_num = 5000;

struct rolling_var_rb_range {
    vector<double> total_sum, total_square_sum, mean, variance;
    vector<int> m_valid_count;
    const int window_size;
    int m_count{0};
    int m_head_index{0};
    std::vector<std::vector<float>> m_container;

    rolling_var_rb_range(int size) : window_size{size + 1} {
        m_container.resize(window_size, std::vector<float>(stock_num, NAN));
        m_valid_count.resize(stock_num, 0);
        total_sum.resize(stock_num, 0);
        total_square_sum.resize(stock_num, 0);
        mean.resize(stock_num, 0);
        variance.resize(stock_num, 0);
    }

    void clear() {
        m_container.clear();
        m_container.resize(window_size, std::vector<float>(stock_num, NAN));
        std::fill(m_valid_count.begin(), m_valid_count.end(), 0);
        std::fill(total_sum.begin(), total_sum.end(), 0);
        std::fill(total_square_sum.begin(), total_square_sum.end(), 0);
        std::fill(mean.begin(), mean.end(), 0);
        std::fill(variance.begin(), variance.end(), 0);
    }

    void delete_old() {
        int old_index = m_head_index - window_size;
        if (old_index < 0) old_index += window_size;
        const vector<float>& old_value = m_container[old_index];
        for (int i = 0; i < stock_num; ++i) {
            if (std::isfinite(old_value[i])) {
                total_sum[i] -= old_value[i];
                total_square_sum[i] -= old_value[i] * old_value[i];
                --m_valid_count[i];
            }
        }
    }
    void add_new() {
        const vector<float>& new_value = m_container[m_head_index - 1];
        for (int i = 0; i < stock_num; ++i) {
            if (std::isfinite(new_value[i])) {
                total_sum[i] += new_value[i];
                total_square_sum[i] += new_value[i] * new_value[i];
                ++m_valid_count[i];
            }

            if (m_valid_count[i] > 1) {
                mean[i] = total_sum[i] / m_valid_count[i];
                variance[i] = (total_square_sum[i] - mean[i] * mean[i] * m_valid_count[i]) / (m_valid_count[i] - 1);
            } else {
                mean[i] = NAN;
                variance[i] = NAN;
            }
        }
    }
    const vector<double>& operator()(const vector<float>& data) {
        m_container[m_head_index++] = data;
        ++m_count;

        if (m_count >= window_size) {
            delete_old();
        }
        add_new();
        if (m_head_index == window_size) m_head_index = 0;
        return variance;
    }
};

struct range_result {
    double total_sum{0}, total_square_sum{0}, mean{0}, variance{0};
    int m_valid_count{0};
};

struct rolling_var_rb_range1 {
    vector<range_result> results;
    const int window_size;
    int m_count{0};
    int m_head_index{0};
    std::vector<std::vector<float>> m_container;

    rolling_var_rb_range1(int size) : window_size{size + 1} {
        m_container.resize(window_size, std::vector<float>(stock_num, NAN));
        results.resize(stock_num);
    }

    void clear() {
        m_container.clear();
        m_container.resize(window_size, std::vector<float>(stock_num, NAN));
        results.clear();
        results.resize(stock_num);
    }

    void delete_old() {
        int old_index = m_head_index - window_size;
        if (old_index < 0) old_index += window_size;
        const vector<float>& old_value = m_container[old_index];
        for (int i = 0; i < stock_num; ++i) {
            range_result& ret = results[i];
            if (std::isfinite(old_value[i])) {
                ret.total_sum -= old_value[i];
                ret.total_square_sum -= old_value[i] * old_value[i];
                --ret.m_valid_count;
            }
        }
    }
    void add_new() {
        const vector<float>& new_value = m_container[m_head_index - 1];
        for (int i = 0; i < stock_num; ++i) {
            range_result& ret = results[i];
            if (std::isfinite(new_value[i])) {
                ret.total_sum += new_value[i];
                ret.total_square_sum += new_value[i] * new_value[i];
                ++ret.m_valid_count;
            }

            if (ret.m_valid_count > 1) {
                ret.mean = ret.total_sum / ret.m_valid_count;
                ret.variance =
                    (ret.total_square_sum - ret.mean * ret.mean * ret.m_valid_count) / (ret.m_valid_count - 1);
            } else {
                ret.mean = NAN;
                ret.variance = NAN;
            }
        }
    }
    void operator()(const vector<float>& data) {
        m_container[m_head_index++] = data;
        ++m_count;

        if (m_count >= window_size) {
            delete_old();
        }
        add_new();
        if (m_head_index == window_size) m_head_index = 0;
    }
};

class MyFixture : public benchmark::Fixture {
public:
    MyFixture() = default;

    ~MyFixture() override = default;

    void SetUp(const ::benchmark::State& state) override {
        window_size = state.range(0);
        int64_t _count = state.range(1);
        data = generate_uniform_matrix(_count, 5000, -1.0f, 1.0f);
        rvrbs.clear();
        rvrbs.resize(5000, rolling_variance_rb(window_size));
        rvrr = new rolling_var_rb_range(window_size);
        rvrr1 = new rolling_var_rb_range1(window_size);
    }

    void TearDown(const ::benchmark::State& state) override {
        rvrbs.clear();
        delete rvrr;
        delete rvrr1;
    }

    double single_test() {
        double ret = 0;
        for (const auto& d : data) {
            for (int i = 0; i < stock_num; ++i) {
                ret += rvrbs[i](d[i]);
            }
        }
        return ret;
    }

    double range_test() {
        double ret = 0;
        for (const vector<float>& d : data) {
            const auto& r = (*rvrr)(d);
            for (int i = 0; i < stock_num; ++i) {
                ret += r[i];
            }
        }
        return ret;
    }

    double range1_test() {
        double ret = 0;
        for (const vector<float>& d : data) {
            (*rvrr1)(d);
            for (int i = 0; i < stock_num; ++i) {
                ret += (*rvrr1).results[i].variance;
            }
        }
        return ret;
    }

    vector<rolling_variance_rb> rvrbs;
    rolling_var_rb_range* rvrr{nullptr};
    rolling_var_rb_range1* rvrr1{nullptr};
    int window_size{5};
    vector<vector<float>> data;
};

BENCHMARK_DEFINE_F(MyFixture, single_test__)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(single_test());
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, single_test__)
    ->Args({5, 1 << 10})
    ->Args({5, 1 << 12})
    ->Args({5, 1 << 14})
    ->Args({5, 1 << 16})
    ->Args({100, 1 << 10})
    ->Args({100, 1 << 12})
    ->Args({100, 1 << 14})
    ->Args({100, 1 << 16});

BENCHMARK_DEFINE_F(MyFixture, range_test__)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(range_test());
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, range_test__)
    ->Args({5, 1 << 10})
    ->Args({5, 1 << 12})
    ->Args({5, 1 << 14})
    ->Args({5, 1 << 16})
    ->Args({100, 1 << 10})
    ->Args({100, 1 << 12})
    ->Args({100, 1 << 14})
    ->Args({100, 1 << 16});

BENCHMARK_DEFINE_F(MyFixture, range1_test__)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(range1_test());
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, range1_test__)
    ->Args({5, 1 << 10})
    ->Args({5, 1 << 12})
    ->Args({5, 1 << 14})
    ->Args({5, 1 << 16})
    ->Args({100, 1 << 10})
    ->Args({100, 1 << 12})
    ->Args({100, 1 << 14})
    ->Args({100, 1 << 16});

BENCHMARK_MAIN();
