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

namespace std {
  template<typename T,typename A> class __attribute__((annotate("vector"))) vector;
  template<typename T,typename A> class __attribute__((annotate("list"))) list;
  template<typename K,typename T,typename C,typename A> class __attribute__((annotate("map"))) map;
  
  template<typename Ch,typename Tr,typename A> class basic_string;
  template<typename T> class char_traits;
  template<typename T> class allocator;
  typedef basic_string<char,std::char_traits<char>,std::allocator<char>> string __attribute__((annotate("string"))) ;
}

.rawInput 0

#include "Def.h"
#include "Spc.h"
#include <vector>
#include <list>
#include <map>
#include <string>
//expected-no-diagnostics
.q
