#ifndef HPC_TIMER_H
#define HPC_TIMER_H

#include <sys/time.h>
#include <zerg_time.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace flux {

class Timer {
public:
    Timer(const std::string& name) : name_(name), start_(ztool::ntime()) {}

    ~Timer() {
        uint64_t elapsed = ztool::ntime() - start_;
        std::cout << name_ << ": " << std::fixed << std::setprecision(9) << ((double)elapsed / 1e9) << " s"
                  << std::endl;
    }

private:
    std::string name_;
    uint64_t start_;
};
}  // namespace flux

#define TIMER(name) flux::Timer timer__(name);

#endif
