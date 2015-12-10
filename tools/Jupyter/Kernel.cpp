//
// Created by Axel Naumann on 09/12/15.
//

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

extern "C" {
///\{
///\name Cling4CTypes
/// The Python compatible view of cling

/// The Interpreter object cast to void*
using TheInterpreter = void ;

/// Create an interpreter object.
TheInterpreter *cling_create(int argc, const char *argv[], const char* llvmdir) {
  auto interp = new cling::Interpreter(argc, argv, llvmdir);
  printf("Interpreter @%p\n", interp);
  return interp;
}

/// Evaluate a string of code. Returns 0 on success.
int cling_eval_dummy(TheInterpreter *interpVP, const char *code) {
  printf("Called cling_eval_dummy\n");
}
/// Evaluate a string of code. Returns 0 on success.
int cling_eval(TheInterpreter *interpVP, const char *code) {
  cling::Interpreter *interp = (cling::Interpreter *) interpVP;
  printf("Interpreter @%p about to run \"%s\"\n", interp, code);
  cling::Value V;
  cling::Interpreter::CompilationResult Res = interp->evaluate(code, V);
  if (Res != cling::Interpreter::kSuccess)
    return 1;
  return 0;
}
///\}

} // extern "C"