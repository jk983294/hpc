## return value optimization
if small value like int, it will return in %eax

if return a big object, it will be put in return slot and pass back its address by %rdi

copy elision rule:
1. (unnamed RVO URVO)return a temporary (prvalue) will trigger copy elision
2. (named RVO NRVO)return a local variable by name
3. even if copy elision doesn't happen, implicit move happens





