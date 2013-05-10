//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUEPRINTER_H
#define CLING_VALUEPRINTER_H

#include "cling/Interpreter/ValuePrinterInfo.h"
#include <string>

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class StoredValueRef;

  // Can be re-implemented to print type-specific details, e.g. as
  //   template <typename POSSIBLYDERIVED>
  //   std::string printValue(const MyClass* const p, POSSIBLYDERIVED* ac,
  //                          const ValuePrinterInfo& VPI);
  template <typename TY>
  std::string printValue(const void* const p, TY* const u,
                         const ValuePrinterInfo& VPI);

  // Can be re-implemented to print a user type differently, e.g. as
  //   template <typename POSSIBLYDERIVED>
  //   std::string printType(const MyClass* const p, POSSIBLYDERIVED* ac,
  //                         const ValuePrinterInfo& VPI);
  template <typename TY>
  std::string printType(const void* const p, TY* const u,
                        const ValuePrinterInfo& VPI);

  namespace valuePrinterInternal {

    std::string printValue_Default(const void* const p,
                                   const ValuePrinterInfo& PVI);
    std::string printType_Default(const ValuePrinterInfo& PVI);

    void StreamStoredValueRef(llvm::raw_ostream& o, const StoredValueRef* VR,
                              clang::ASTContext& C, const char* Sep = "\n");

    void flushToStream(llvm::raw_ostream& o, const std::string& s);

    template <typename T>
    const T& Select(llvm::raw_ostream* o, clang::Expr* E,
                    clang::ASTContext* C, const T& value) {
      ValuePrinterInfo VPI(E, C);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushToStream(*o, printType(&value, &value, VPI)
                    + printValue(&value, &value, VPI));
      return value;
    }

    template <typename T>
    const T* Select(llvm::raw_ostream* o, clang::Expr* E,
                    clang::ASTContext* C, const T* value) {
      ValuePrinterInfo VPI(E, C);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushToStream(*o, printType((const void*) value, value, VPI)
                    + printValue((const void*) value, value, VPI));
      return value;
    }

    template <typename T>
    const T* Select(llvm::raw_ostream* o, clang::Expr* E,
                    clang::ASTContext* C, T* value) {
      ValuePrinterInfo VPI(E, C);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushToStream(*o, printType((const void*) value, value, VPI)
                    + printValue((const void*) value, value, VPI));
      return value;
    }

  } // namespace valuePrinterInternal

  template <typename TY>
  std::string  printValue(const void* const p, TY* const /*u*/,
                          const ValuePrinterInfo& PVI) {
    return valuePrinterInternal::printValue_Default(p, PVI);
  }
  template <typename TY>
  std::string  printType(const void* const p, TY* const /*u*/,
                         const ValuePrinterInfo& PVI) {
    return valuePrinterInternal::printType_Default(PVI);
  }

}

#endif // CLING_VALUEPRINTER_H
