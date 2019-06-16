#include <omp.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

double calc_a(int i) { return 1.1 * i; }
double consume(double i) { return i * 2; }

int main() {
    omp_set_num_threads(4);
    int len = 1000;
    double result = 0;

    #pragma omp parallel num_threads(4)
    {
        int id = omp_get_thread_num();
        int actual_thread_count = omp_get_num_threads();
        for (int i = id; i < len; i += actual_thread_count) {  // round robin
            double x = calc_a(i);

            /**
             * critical means critical section, execute exclusively
             * atomic means if hardware support below statements mutual exclusively, use it, otherwise work as critical
             */
            #pragma omp atomic
            result += x;
//          #pragma omp critical
//          {
//              result += x;
//          }
        }
    }

    cout << result << endl;
    cout << "done " << endl;
    return 0;
}
