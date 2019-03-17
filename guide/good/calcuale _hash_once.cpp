#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

struct Data {
    int data;
};

Data* getDataBad(const string& key, unordered_map<string, std::unique_ptr<Data>>& cache) {
    if (cache[key]) {
        return cache[key].get();
    }

    cache[key] = std::make_unique<Data>();
    return cache[key].get();
}

/**
 * calculate hash only once
 */
Data* getDataGood(const string& key, unordered_map<string, std::unique_ptr<Data>>& cache) {
    std::unique_ptr<Data>& entry = cache[key];
    if (entry) {
        return entry.get();
    }

    entry = std::make_unique<Data>();
    return entry.get();
}

int main() {
    cout << "hello" << endl;
    return 0;
}
