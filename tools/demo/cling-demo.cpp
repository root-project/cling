#include <cling/Interpreter/Interpreter.h>
#include <cling/Interpreter/Value.h>
#include <iostream>
#include <string>
#include <sstream>

///\brief Pass a value from the compiled program into cling, let cling modify it
///  and return the modified value into the compiled program.
int setAndUpdate(int arg, cling::Interpreter& interp) {
   int ret = arg; // The value that will be modified
                  // Update the value of ret by passing it to the interpreter.
   std::ostringstream sstr;
   sstr << "int& ref = *(int*)" << &ret << ';';
   sstr << "ref = ref * ref;";
   interp.process(sstr.str());
   return ret;
}

///\brief A ridiculously complicated way of converting an int to a string.
std::unique_ptr<std::string> stringify(int value, cling::Interpreter& interp) {

   // Declare the function to cling:
   static const std::string codeFunc = R"CODE(
#include <string>
std::string* createAString(const char* str) {
   return new std::string(str);
})CODE";
   interp.declare(codeFunc);

   // Call the function with "runtime" values:
   std::ostringstream sstr;
   sstr << "createAString(\"" << value << "\");";
   cling::Value res; // Will hold the result of the expression evaluation.
   interp.process(sstr.str(), &res);

   // Grab the return value of `createAString()`:
   std::string* resStr = reinterpret_cast<std::string*>(res.getPtr());
   return std::unique_ptr<std::string>(resStr);
}

int main(int argc, const char* const* argv) {
    // Create the Interpreter. LLVMDIR is provided as -D during compilation.
    cling::Interpreter interp(argc, argv, LLVMDIR);

    std::cout << setAndUpdate(17, interp) << '\n';
    std::cout << *stringify(42, interp) << '\n';

    return 0;
}
