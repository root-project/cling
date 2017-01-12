//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify

// Check that decl extraction doesn't complain about unrelated decls
.rawInput 1
namespace UNRELATED { void name(); }
using namespace UNRELATED;
.rawInput 0
int name = 12;

// Check that decl extraction doesn't complain about unrelated decls
.rawInput 1
namespace N { void injected(); } // expected-note {{target of using declaration}}
using N::injected; // expected-note {{using declaration}}
.rawInput 0
int injected = 13; // expected-error {{declaration conflicts with target of using declaration already in scope}}

// Check that decl extraction does complain about clashing decls
extern "C" double likeSin(double); // expected-note {{previous definition is here}}
int likeSin = 14; // expected-error {{redefinition of 'likeSin' as different kind of symbol}}

// Test a weakness in the declaration extraction of types (ROOT-5248).
class MyClass; // this type...
extern MyClass* my;
class MyClass { // and that type used to not be redecls
public:
  MyClass* getMyClass() {
    return 0;
  }
} cl;
MyClass* my = cl.getMyClass();

.q
