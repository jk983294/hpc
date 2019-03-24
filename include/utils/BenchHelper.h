#ifndef HPC_BENCH_HELPER_H
#define HPC_BENCH_HELPER_H

#include <zerg_file.h>
#include <zerg_string.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

using namespace std;

namespace flux {

const char* RandomStringFileName = "/tmp/benchmark.random.strings";

inline vector<string> GenRandomStrings(int64_t targetCount, size_t len = 8) {
    vector<string> result;
    int count = 0;
    unordered_set<string> s;
    while (count < targetCount) {
        string randStr = ztool::GenerateRandomString(len);
        if (s.find(randStr) == s.end()) {
            s.insert(randStr);
            result.push_back(randStr);
            ++count;
        }
    }
    return result;
}

inline void GenRandomStringFile(int64_t targetCount = 1000000) {
    if (ztool::IsFileExisted(RandomStringFileName)) return;

    vector<string> strings = GenRandomStrings(targetCount);
    ofstream ofs(RandomStringFileName, ofstream::out | ofstream::trunc);

    if (!ofs) {
        cerr << "open file failed" << endl;
    } else {
        for (const auto& str : strings) {
            ofs << str << endl;
        }
        ofs.close();
        cout << "GenRandomStringFile success" << endl;
    }
}

inline vector<string> GetRandomStringFromFile() {
    vector<string> strings;
    if (ztool::IsFileExisted(RandomStringFileName)) {
        ifstream ifs(RandomStringFileName, ifstream::in);
        if (ifs.is_open()) {
            string s;
            while (getline(ifs, s)) {
                strings.push_back(s);
            }
            ifs.close();
        }
    }
    cout << "get " << strings.size() << " entry from file" << endl;
    return strings;
}
}  // namespace flux

#endif
