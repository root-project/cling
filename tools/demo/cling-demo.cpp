//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include <cling/Interpreter/Interpreter.h>
#include <cling/Interpreter/Value.h>
#include <cling/Utils/Casting.h>
#include <iostream>
#include <string>
#include <sstream>

/// Definitions of declarations injected also into cling.
/// NOTE: this could also stay in a header #included here and into cling, but
/// for the sake of simplicity we just redeclare them here.
int aGlobal = 42;
static float anotherGlobal = 3.141;
float getAnotherGlobal() { return anotherGlobal; }
void setAnotherGlobal(float val) { anotherGlobal = val; }

///\brief Call compiled functions from the interpreter.
void useHeader(cling::Interpreter& interp) {
  // We could use a header, too...
  interp.declare("int aGlobal;\n"
                 "float getAnotherGlobal();\n"
                 "void setAnotherGlobal(float val);\n");

  cling::Value res; // Will hold the result of the expression evaluation.
  interp.process("aGlobal;", &res);
  std::cout << "aGlobal is " << res.getAs<long long>() << '\n';
  interp.process("getAnotherGlobal();", &res);
  std::cout << "getAnotherGlobal() returned " << res.getAs<float>() << '\n';

  setAnotherGlobal(1.); // We modify the compiled value,
  interp.process("getAnotherGlobal();", &res); // does the interpreter see it?
  std::cout << "getAnotherGlobal() returned " << res.getAs<float>() << '\n';

  // We modify using the interpreter, now the binary sees the new value.
  interp.process("setAnotherGlobal(7.777); getAnotherGlobal();");
  std::cout << "getAnotherGlobal() returned " << getAnotherGlobal() << '\n';
}

///\brief Call an interpreted function using its symbol address.
void useSymbolAddress(cling::Interpreter& interp) {
  // Declare a function to the interpreter. Make it extern "C" to remove
  // mangling from the game.
  interp.declare("extern \"C\" int plutification(int siss, int sat) "
                 "{ return siss * sat; }");
  void* addr = interp.getAddressOfGlobal("plutification");
  using func_t = int(int, int);
  func_t* pFunc = cling::utils::VoidToFunctionPtr<func_t*>(addr);
  std::cout << "7 * 8 = " << pFunc(7, 8) << '\n';
}

///\brief Pass a pointer into cling as a string.
void usePointerLiteral(cling::Interpreter& interp) {
  int res = 17; // The value that will be modified

  // Update the value of res by passing it to the interpreter.
  std::ostringstream sstr;
  // on Windows, to prefix the hexadecimal value of a pointer with '0x',
  // one need to write: std::hex << std::showbase << (size_t)pointer
  sstr << "int& ref = *(int*)" << std::hex << std::showbase << (size_t)&res << ';';
  sstr << "ref = ref * ref;";
  interp.process(sstr.str());
  std::cout << "The square of 17 is " << res << '\n';
}

int main(int argc, const char* const* argv) {
  // Create the Interpreter. LLVMDIR is provided as -D during compilation.
  cling::Interpreter interp(argc, argv, LLVMDIR);

  useHeader(interp);
  useSymbolAddress(interp);
  usePointerLiteral(interp);

  return 0;
}
