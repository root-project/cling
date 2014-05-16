//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUEPRINTER_H
#define CLING_VALUEPRINTER_H

#include "cling/Interpreter/Value.h"
#include <string>

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class Value;

  ///\brief Generic interface to value printing.
  ///
  /// Can be re-implemented to print type-specific details, e.g. as
  ///\code
  ///   template <typename POSSIBLYDERIVED>
  ///   std::string printValue(const MyClass* const p, POSSIBLYDERIVED* ac,
  ///                          const Value& V);
  ///\endcode
  //template <typename TY>
  //std::string printValue(const void* const p, TY* const u,
  //                       const Value& V);

  ///\brief Generic interface to type printing.
  ///
  /// Can be re-implemented to print a user type differently, e.g. as
  ///\code
  ///   template <typename POSSIBLYDERIVED>
  ///   std::string printType(const MyClass* const p, POSSIBLYDERIVED* ac,
  ///                         const Value& V);
  ///\endcode
  //template <typename TY>
  //std::string printType(const void* const p, TY* const u,
  //                      const Value& V);

  namespace valuePrinterInternal {

    std::string printValue_Default(const Value& V);
    std::string printType_Default(const Value& V);

    void StreamClingValue(llvm::raw_ostream& o, const Value* VR);

    void flushToStream(llvm::raw_ostream& o, const std::string& s);

    template<typename T>
    void Select(llvm::raw_ostream* o, const Value* value) {
      // Only because we don't want to include llvm::raw_ostream in the header
      //flushToStream(*o, printType(0, (void*)0, value)
      //              + printValue(0, (void*)0, value) + '\n');
    }
 
    /*    template <typename T>
    const T& Select(llvm::raw_ostream* o, Interpreter* I,
              clang::ASTContext* C, const T& value) {
      ValuePrinterInfo VPI(I, C);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushToStream(*o, printType(&value, &value, VPI)
                    + printValue(&value, &value, VPI) + '\n');
      return value;
    }

    template <typename T>
    T* Select(llvm::raw_ostream* o, Interpreter* I,
                    clang::ASTContext* C, T* value) {
      ValuePrinterInfo VPI(I, C);
      // Only because we don't want to include llvm::raw_ostream in the header
      flushToStream(*o, printType((const void*) value, value, VPI)
                    + printValue((const void*) value, value, VPI) + '\n');
      return value;
    }
    */
  } // namespace valuePrinterInternal

  ///\brief Catch-all implementation for value printing.

}

#endif // CLING_VALUEPRINTER_H
