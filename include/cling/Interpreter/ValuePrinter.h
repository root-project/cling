//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUEPRINTER_H
#define CLING_VALUEPRINTER_H

#include "cling/Interpreter/ValuePrinterInfo.h"

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class StoredValueRef;

  // Can be re-implemented to print type-specific details, e.g. as
  //   template <typename ACTUAL>
  //   void dumpPtr(llvm::raw_ostream& o, const clang::Decl* a, ACTUAL* ac,
  //                int flags, const char* tname);
  template <typename TY>
  void printValue(llvm::raw_ostream& o, const void* const p,
                  TY* const u, const ValuePrinterInfo& VPI);

  namespace valuePrinterInternal {

    void printValue_Default(llvm::raw_ostream& o, const void* const p,
                            const ValuePrinterInfo& PVI);

    void StreamStoredValueRef(llvm::raw_ostream& o, const StoredValueRef* VR,
                              clang::ASTContext& C, const char* Sep = "\n");

    void flushOStream(llvm::raw_ostream& o);

    template <typename T>
    const T& Select(llvm::raw_ostream* o, clang::Expr* E,
                    clang::ASTContext* C, const T& value) {
      ValuePrinterInfo VPI(E, C);
      printValue(*o, &value, &value, VPI);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushOStream(*o);
      return value;
    }

    template <typename T>
    const T* Select(llvm::raw_ostream* o, clang::Expr* E,
                    clang::ASTContext* C, const T* value) {
      ValuePrinterInfo VPI(E, C);
      printValue(*o, (const void*) value, value, VPI);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushOStream(*o);
      return value;
    }

    template <typename T>
    const T* Select(llvm::raw_ostream* o, clang::Expr* E,
                    clang::ASTContext* C, T* value) {
      ValuePrinterInfo VPI(E, C);
      printValue(*o, (const void*) value, value, VPI);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushOStream(*o);
      return value;
    }

  } // namespace valuePrinterInternal


  // Can be re-implemented to print type-specific details, e.g. as
  //   template <typename ACTUAL>
  //   void dumpPtr(llvm::raw_ostream& o, const clang::Decl* a,
  //                ACTUAL* ap, int flags, const char* tname);
  template <typename TY>
  void printValue(llvm::raw_ostream& o, const void* const p,
                  TY* const u, const ValuePrinterInfo& PVI) {
    valuePrinterInternal::printValue_Default(o, p, PVI);
  }

}

#endif // CLING_VALUEPRINTER_H
