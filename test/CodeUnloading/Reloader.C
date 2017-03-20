//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --noruntime -DCLING_NO_NORUTIME -I%S -Xclang -verify 2>&1 | FileCheck %s
// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s
// XFAIL: *
// Changes to CodeGenModule::Release() breaks this test:
//     DeferredDecls.insert(EmittedDeferredDecls.begin(),
//                          EmittedDeferredDecls.end());

// Test reload-differing-layouts

extern "C" int printf(const char*, ...);

// FIXME: Would be nice to have FileCheck verify type-name matches between lines
#define DUMP_TYPE typeid(std::string).name()
#include <typeinfo>

#include <string>
printf("%s %s\n", std::string("std::string").c_str(), DUMP_TYPE);
.undo
.undo
// CHECK: std::string

#include "Reloader.h"
printf("%s %s\n", std::string("local::string").c_str(), DUMP_TYPE);
.undo
.undo
// CHECK-NEXT: local::string

#include <string>
printf("%s %s\n", std::string("std::string(2)").c_str(), DUMP_TYPE);
.undo
.undo
// CHECK-NEXT: std::string(2)

.undo // #include <typeinfo>

#ifdef CLING_NO_NORUTIME
 #define POSE_AS_STD_NAMESPACE_ namespace std { inline namespace __1 {
 #define _POSE_AS_STD_NAMESPACE } }
#endif

#define POSE_NOT_TEMPLATED
#include "Reloader.h"

printf("%s\n", std::string("local::string(2)").c_str());
.undo
.undo
// CHECK-NEXT: local::string(2)

// expected-no-diagnostics
.q
