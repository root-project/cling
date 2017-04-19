//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

#include "cling/Interpreter/Interpreter.h"
#include <vector>
#include <string>

std::vector<std::string> completions;
std::string input = "gCling";
size_t point = 3;
gCling->codeComplete(input, point, completions);
//completions 

// globalVar<TAB> expected globalVariable
int globalVariable;
completions.clear();
input = "globalVariable"
point = 7;
gCling->codeComplete(input, point, completions); //CHECK: (std::basic_string &) "globalVariable"



.rawInput 1

struct MyStruct {
  MyStruct() {}
  MyStruct anOverload(int) { return MyStruct(); }
  MyStruct anOverload(float) { return MyStruct(); }
};

.rawInput 0

// MyStr<TAB> m;
completions.clear();
input = "MyStruct"
point = 5;
gCling->codeComplete(input, point, completions); 
completions //CHECK: (std::vector<std::string> &) { "MyStruct" }

MyStruct m;
// m.<TAB>
completions.clear();
input = "m.";
point = 2;
gCling->codeComplete(input, point, completions); 
completions //CHECK: (std::vector<std::string> &) { "[#MyStruct#]anOverload(<#int#>)", "[#MyStruct#]anOverload(<#float#>)", "MyStruct::", "[#MyStruct &#]operator=(<#const MyStruct &#>)", "[#MyStruct &#]operator=(<#MyStruct &&#>)" }

// an<TAB>K(12)
completions.clear();
input = "m.an";
point = 4;
gCling->codeComplete(input, point, completions);
completions //CHECK: (std::vector<std::string> &) { "[#MyStruct#]anOverload(<#int#>)", "[#MyStruct#]anOverload(<#float#>)" }



.rawInput 1

extern "C" int printf(const char* fmt, ...);

namespace MyNamespace {
  class MyClass {
  public:
    MyClass() { printf("MyClass constructor called!\n"); }
  };

  void f() {
    printf("Function f in namespace MyNamespace called!\n");
  }
}

.rawInput 0

//My<TAB> // expected MyNamespace, MyClass
completions.clear();
input = "My";
point = 2;
gCling->codeComplete(input, point, completions); 
completions //CHECK: (std::vector<std::string> &) { "MyNamespace::", "MyStruct" }

//MyNames<TAB> // expected MyNamespace //expected MyClass, f
completions.clear();
input = "MyNames";
point = 7;
gCling->codeComplete(input, point, completions); 
completions //CHECK: (std::vector<std::string> &) { "MyNamespace::" }

//MyNames<TAB> // expected MyNamespace //expected MyClass, f
completions.clear();
input = "MyNamespace::";
point = 13;
gCling->codeComplete(input, point, completions); 
completions //CHECK: (std::vector<std::string> &) { "[#void#]f()", "MyClass" }

//MyNamespace::MyC<TAB> // expected MyClass
completions.clear();
input = "MyNamespace::MyC";
point = 16;
gCling->codeComplete(input, point, completions); 
completions //CHECK: (std::vector<std::string> &) { "MyClass" }
