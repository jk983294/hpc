#include <omp.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

double x{1.1}, y{2.2}, z{3.3};

void calc_x() { x *= 2; }
void calc_y() { y *= 2; }
void calc_z() { z *= 2; }

int main() {
    omp_set_num_threads(4);
    cout << x << ' ' << y << ' ' << z << ' ' << endl;

    #pragma omp parallel num_threads(3)
    {
        /**
         * for section in sections, one thread pick one section, no sync is guaranteed
         */
        #pragma omp sections
        {
            #pragma omp section
            calc_x();
            #pragma omp section
            calc_y();
            #pragma omp section
            calc_z();
        }
    }

    cout << x << ' ' << y << ' ' << z << ' ' << endl;
    cout << "done" << endl;
    return 0;
}
