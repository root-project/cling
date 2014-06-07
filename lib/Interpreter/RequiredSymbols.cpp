//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"
//#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/DynamicExprInfo.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/AutoloadCallback.h"
#include "clang/AST/Type.h"

#include "llvm/Support/raw_ostream.h"

namespace cling {
namespace internal {
void symbol_requester() {
   const char* const argv[] = {"libcling__symbol_requester", 0};
   Interpreter I(1, argv);
   //cling_PrintValue(0);
   LookupHelper h(0,0);
   h.findType("", LookupHelper::NoDiagnostics);
   h.findScope("", LookupHelper::NoDiagnostics);
   h.findFunctionProto(0, "", "", LookupHelper::NoDiagnostics);
   h.findFunctionArgs(0, "", "", LookupHelper::NoDiagnostics);
   runtime::internal::DynamicExprInfo DEI(0,0,false);
   DEI.getExpr();
   InterpreterCallbacks cb(0);
   AutoloadCallback a(&I);
}
}
}
