//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: Interpreter.cpp 44226 2012-05-11 16:41:08Z vvassilev $
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Callable.h"

#include "llvm/Function.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

cling::Callable::Callable(const clang::FunctionDecl& Decl,
                          const cling::Interpreter& Interp):
  decl(&Decl),
  func(0),
  exec(Interp.getExecutionEngine())
{
  if (exec) {
    std::string mangledName;
    llvm::raw_string_ostream RawStr(mangledName);
    llvm::OwningPtr<clang::MangleContext>
      Mangle(Interp.getCI()->getASTContext().createMangleContext());
    Mangle->mangleName(decl, RawStr);
    RawStr.flush();
    func = exec->FindFunctionNamed(mangledName.c_str());
  }
}


bool cling::Callable::Invoke(const std::vector<llvm::GenericValue>& ArgValues,
                             Value* Result /*= 0*/) const
{
  if (!isValid()) return false;
  if (Result) {
    *Result = Value(exec->runFunction(func, ArgValues),
                    decl->getCallResultType());
  } else {
    exec->runFunction(func, ArgValues);
  }
  return true;
}
