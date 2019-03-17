#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace std;

/**
 * for static local variable initialization, you have a guard variable to check if initialized
 * 5% - 10% overhead compared to non-static access, even it is single threaded
 */

struct Random {
    int get() {
        static int i = rand();
        return i;
    }
};

// Random::get():
//        push    rbp
//        mov     rbp, rsp
//        sub     rsp, 16
//        mov     QWORD PTR [rbp-8], rdi
//        movzx   eax, BYTE PTR guard variable for Random::get()::i[rip]
//        test    al, al
//        sete    al
//        test    al, al
//        je      .L2
//        mov     edi, OFFSET FLAT:guard variable for Random::get()::i
//        call    __cxa_guard_acquire
//        test    eax, eax
//        setne   al
//        test    al, al
//        je      .L2
//        call    rand
//        mov     DWORD PTR Random::get()::i[rip], eax
//        mov     edi, OFFSET FLAT:guard variable for Random::get()::i
//        call    __cxa_guard_release
//.L2:
//        mov     eax, DWORD PTR Random::get()::i[rip]
//        leave
//        ret

int main() {
    Random r;
    return r.get();
}
