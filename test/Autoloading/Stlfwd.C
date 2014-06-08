//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// Test stlFwd
//XFAIL: *
//fail because the way the autoloading transformation is loaded now, causes assertion failure for this
namespace std {
    
  template <typename T,typename A> class __attribute__((annotate("vector"))) vector;
  template <typename T,typename A> class __attribute__((annotate("list"))) list;
  template <typename K,typename T,typename C,typename A> class __attribute__((annotate("map"))) map;
  
  template <typename Ch,typename Tr,typename A> class basic_string;
  template <typename T> class char_traits;
  template <typename T> class allocator;
  typedef basic_string<char,char_traits<char>,allocator<char>> string __attribute__((annotate("string"))) ;
  
  template <typename R> void sort(R,R) __attribute__((annotate("algorithm")));
  template <typename R,typename C> void sort(R,R,C) __attribute__((annotate("algorithm")));
  
  template< bool B, typename T> struct __attribute__((annotate("type_traits"))) enable_if;
}

#include<vector>
#include<list>
#include<type_traits>
#include<map>
#include<algorithm>
#include<string>