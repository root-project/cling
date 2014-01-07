//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// This file contains an error (redefinition of '__my_i') and it gets included
// so all the contents should be reverted from the AST transparently.

// Template specializations
template<> int TemplatedF(int t){return t + 10;}
template<> double TemplatedF(double t){return t + 3.14;}

// Aliases
typedef struct A AStruct;


// Overloads
int OverloadedF(int i){ return i + 10;};

// Redeclarations
int __my_i = 0; // expected-note {{previous definition is here}}
int __my_i = 0; // expected-error {{redefinition of '__my_i'}}
