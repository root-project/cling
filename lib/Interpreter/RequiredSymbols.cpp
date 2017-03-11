//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/DynamicExprInfo.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Utils/Output.h"
//#include "cling-c/ValuePrinter.h"
#include "cling-c/Exception.h"

#include "clang/AST/Type.h"

namespace cling {
namespace internal {
void symbol_requester() {
   const char* const argv[] = {"libcling__symbol_requester", 0};
   Interpreter I(1, argv);
   //cling_PrintValue(0);
   // sharedPtr is needed for SLC6 with devtoolset2:
   // Redhat re-uses libstdc++ from GCC 4.4 and patches missing symbols into
   // the binary through an archive. We need to pull more symbols from the
   // archive to make them available to cling. This list will possibly need to
   // grow...
   std::shared_ptr<int> sp;
   Interpreter* SLC6DevToolSet = (Interpreter*)(void*)&sp;
   LookupHelper h(0,SLC6DevToolSet);
   h.findType("", LookupHelper::NoDiagnostics);
   h.findScope("", LookupHelper::NoDiagnostics);
   h.findFunctionProto(0, "", "", LookupHelper::NoDiagnostics);
   h.findFunctionArgs(0, "", "", LookupHelper::NoDiagnostics);
   runtime::internal::DynamicExprInfo DEI(0,0,false);
   DEI.getExpr();
   cling_ThrowIfInvalidPointer(nullptr, nullptr, nullptr);
}

// test/ErrorRecovery/Exceptions.C calls this function so to make sure the
// stack is properly unwound when throwing from the JIT or null deref checking
//
void TestUnwind(void (*Proc)(intptr_t), intptr_t Data) {
  if (Proc) {
    struct Unwinder {
      ~Unwinder() {
        printf("UnwindTest::~UnwindTest\n");
        fflush(stdout);
      }
    };
    Unwinder UW;
    Proc(Data);
  } else {
    // If TestUnwind isn't stripped, why would symbol_requester?
    // Whatever...Prepare to crash!
    symbol_requester();
  }
}

}
}
