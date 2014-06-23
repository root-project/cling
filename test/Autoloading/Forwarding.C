//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// Test forwardDeclaration

.rawInput 1
int id(int) __attribute__((annotate("Def.h")));

class __attribute__((annotate("Def.h"))) C;

namespace N {
  void nested() __attribute__((annotate("Def.h")));
}

template <typename T> class __attribute__((annotate("Def.h"))) Gen;
template <> class __attribute__((annotate("Def.h"))) Gen<int> ;
template <> class __attribute__((annotate("Spc.h"))) Gen<float>;

template <typename T,typename U> class  __attribute__((annotate("Def.h"))) Partial;
template <typename T> class __attribute__((annotate("Spc.h"))) Partial<T,int>;

#include "Def.h"
#include "Spc.h"

//expected-no-diagnostics
.q
