namespace test {
template <bool B, class T, class F> struct __attribute__((annotate("./Fail.h")))  conditional;
;
template <typename _Tp> using example __attribute__((annotate("./Fail.h")))  = typename conditional<true, int, float>::type;
}
;
namespace test {
class __attribute__((annotate("./Fail.h")))  HasSubType;
class HasSubType::SubType FunctionUsingSubtype(class HasSubType::SubType s) __attribute__((annotate("./Fail.h")))  ;
}
;
namespace test {
typedef long ssize_t __attribute__((annotate("./Fail.h"))) ;
typedef unsigned int size_t __attribute__((annotate("./Fail.h"))) ;
extern ssize_t FunctionUsingRestrictPtr(const char *restrict __path, char *restrict __buf, size_t __len) __attribute__((annotate("./Fail.h"))) ;
}
;
namespace test {
template <typename T, int MAX> class __attribute__((annotate("./Fail.h")))  Stack;
Stack<int> FunctionReturningStack() __attribute__((annotate("./Fail.h")))  ;
}
;
