#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

/**
 * std::function may allocate
 * check SG14's inplace_function.h
 */

struct Point {
    double dimensions[3];
};

// main:
//        push    rbx
//        mov     edi, 24
//        sub     rsp, 48
//        mov     QWORD PTR [rsp+32], 0
//        call    operator new(unsigned long)
//        pxor    xmm0, xmm0
//        lea     rsi, [rsp+16]
//        mov     edx, 3
int main() {
    std::function<void()> no_op{[point = Point{}] {}};
    return 0;
}
