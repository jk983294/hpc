#include <omp.h>
#include <utils/BenchHelper.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

constexpr int len = 10000000;
constexpr int thread_count = 4;

int main() {
    omp_set_num_threads(4);

    vector<double> data;
    int flag = 0, tmp_flag = 0;
    double result = 0.0;

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            data = flux::GenRandomNumbers(0.0, len * 1.0, len);
            #pragma omp flush  // flush data

            #pragma omp atomic write
                flag = 1;
            #pragma omp flush(flag)  // flush only flag to notify consumer
        }
        #pragma omp section
        {
            while (1) {
                #pragma omp flush(flag)  // get latest flag and check
                #pragma omp atomic write
                    tmp_flag = flag;
                if (tmp_flag == 1) break;
            }

            #pragma omp flush  // need to flush to get data visible
            result = std::accumulate(data.begin(), data.end(), 0.0);
        }
    }

    cout << result << endl;
    cout << "done" << endl;
    return 0;
}
