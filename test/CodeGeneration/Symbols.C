//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: clang -shared -fPIC -DBUILD_SHARED %s -o%T/libSymbols%shlibext
// RUN: %cling --nologo -L%T -lSymbols  %s | FileCheck %s

// Check that weak symbols do not get re-emitted (ROOT-6124)
extern "C" int printf(const char*,...);

template <class T>
struct StaticStuff {
  static T s_data;
};
template <class T>
T StaticStuff<T>::s_data = 42;

int compareAddr(int* interp);
#ifdef BUILD_SHARED
int compareAddr(int* interp) {
  if (interp != &StaticStuff<int>::s_data) {
    printf("Wrong address, %ld in shared lib, %ld in interpreter!\n",
           (long)&StaticStuff<int>::s_data,
           (long)interp);
    // CHECK-NOT: Wrong address
    return 1;
  }
  return 0;
}
#else
int Symbols() {
  return compareAddr(&StaticStuff<int>::s_data);
}
#endif

