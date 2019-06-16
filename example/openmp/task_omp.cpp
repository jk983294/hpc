#include <omp.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

struct Node {
    int data{0};
    Node* next{nullptr};

    Node(int x) : data{x} {}
    Node() = default;
};

constexpr int len = 10000000;
constexpr int thread_count = 4;

void process(Node* node) { node->data++; }

int main() {
    omp_set_num_threads(4);

    Node head;
    Node* current = &head;
    for (int i = 0; i < len; ++i) {
        current->next = new Node(i);
        current = current->next;
    }
    Node* last = current;
    cout << "current last " << last->data << endl;

    #pragma omp parallel num_threads(thread_count)
    {
        /**
         * the idea is one thread will go through the list and create task for each node
         * the rest threads will execute those tasks created by that thread
         */
        #pragma omp single
        {
            current = head.next;
            while (current) {
                #pragma omp task firstprivate(current)
                { process(current); }

                current = current->next;
            }
        }
    }

    cout << "current last " << last->data << endl;
    cout << "done " << endl;
    return 0;
}
