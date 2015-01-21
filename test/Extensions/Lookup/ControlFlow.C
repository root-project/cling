//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p | FileCheck %s

// The tests shows the basic control flow structures that contain dynamic
// expressions. There are several cases that could be distinguished.
// 1. IfStmt with dynamic expression in the condition like if (h->Draw())
// In that case we know that the condition is going to be implicitly
// casted to bool. Clang does it even for integral types.
//   1.1 if the dynamic expression result type is bool no cast is needed
//       and there shouldn't be problems at all.
//   1.2 if the dynamic expression is of integral type like int clang
//       makes implicit cast. I've tested with few examples and it works
//       even without the cast
// The real problem is that if we want to stick to clang's AST in the same
// way as clang builds it we need to have implicit casts for the dynamic
// expressions which return type differs from bool. The problem is we cannot
// do that at compile time because we don't know the return type of the
// expression.


#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
extern "C" int printf(const char*,...);

.dynamicExtensions
std::unique_ptr<cling::test::SymbolResolverCallback> SRC;
SRC.reset(new cling::test::SymbolResolverCallback(gCling))
gCling->setCallbacks(std::move(SRC));

int a[5] = {1,2,3,4,5};
if (h->PrintArray(a, 5)) { // runtime result type bool
  whatever->PrintString("Replaced in then.\n");
  printf("\n%s\n", "Array Printed Successfully!");
}
// CHECK: 12345
// CHECK: Array Printed Successfully!

int b, c = 1;
// Note that in case of function not returning bool we need an ImplicitCast
// which is tricky because we don't know the type of the function
if (h->Add(b, c)) { // runtime result type int
  printf("\n%s\n", "Sum more than 0!");
}
// CHECK: Sum more than 0!

for (int i = 0; i < 2; ++i) {
  dependentInLoop->Add(b, c);
 }
//CHECK-NOT:(int) 1
.q
