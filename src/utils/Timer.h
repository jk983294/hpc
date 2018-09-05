#ifndef HPC_TIMER_H
#define HPC_TIMER_H

#include <sys/time.h>
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
    Timer(const std::string& name) : name_(name), start_(std::clock()) {}

    ~Timer() {
        double elapsed = (double(std::clock() - start_) / double(CLOCKS_PER_SEC));
        std::cout << name_ << ": " << std::fixed << std::setprecision(9) << elapsed << " s" << std::endl;
    }

private:
    std::string name_;
    std::clock_t start_;
};

static constexpr uint64_t oneSecondNano{1000000000};

inline double timespec2double(const timespec& ts) {
    long usec = ts.tv_nsec / 1000;
    return static_cast<double>(ts.tv_sec) + (static_cast<double>(usec) / 1000000.0);
}

inline void double2timespec(double t, timespec& ts) {
    ts.tv_sec = static_cast<time_t>(t);
    double dt = (t - static_cast<double>(ts.tv_sec)) * 1000000.0;
    dt = floor(dt + 0.5);
    long usec = static_cast<long>(dt);
    ts.tv_nsec = usec * 1000;
}
inline uint64_t timespec2nanos(const timespec& ts) { return ts.tv_sec * oneSecondNano + ts.tv_nsec; }

inline void nanos2timespec(uint64_t t, timespec& ts) {
    ts.tv_sec = t / oneSecondNano;
    ts.tv_nsec = t % oneSecondNano;
}

inline int64_t nanoSinceEpoch() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<int64_t>(ts.tv_sec * oneSecondNano + ts.tv_nsec);
}

inline uint64_t nanoSinceEpochU() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * oneSecondNano + ts.tv_nsec;
}

inline uint64_t ntime() { return nanoSinceEpochU(); }

inline std::string time_t2string(const time_t ct) {
    if (!ct) return "N/A";
    struct tm tm;
    localtime_r(&ct, &tm);
    char buffer[21];
    std::snprintf(buffer, sizeof buffer, "%4u-%02u-%02u %02u:%02u:%02u", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return string(buffer);
}
inline std::string ntime2string(uint64_t nano) {
    time_t epoch = nano / oneSecondNano;
    return time_t2string(epoch);
}

inline timeval now_timeval() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv;
}

inline std::string now_string() {
    time_t tNow = time(NULL);
    struct tm tm;
    localtime_r(&tNow, &tm);
    char buffer[16];
    std::snprintf(buffer, sizeof buffer, "%4u%02u%02u.%02u%02u%02u", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return string(buffer);
}
}

#define TIMER(name) flux::Timer timer__(name);

#endif
