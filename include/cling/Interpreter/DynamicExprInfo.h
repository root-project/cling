//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DYNAMIC_EXPR_INFO_H
#define CLING_DYNAMIC_EXPR_INFO_H

#include <string>

namespace cling {
namespace runtime {
  namespace internal {
    ///\brief Helper structure used to provide specific context of the evaluated
    /// expression, when needed.
    ///
    /// Consider:
    /// @code
    /// int a = 5;
    /// const char* b = dep->Symbol(a);
    /// @endcode
    /// In the particular case we need to pass a context to the evaluator of the
    /// unknown symbol. The addresses of the items in the context are not known at
    /// compile time, so they cannot be embedded directly. Instead of that we
    /// need to create an array of addresses of those context items (mainly
    /// variables) and insert them into the evaluated expression at runtime
    /// This information is kept using the syntax: "dep->Symbol(*(int*)@)",
    /// where @ denotes that the runtime address the variable "a" is needed.
    ///
    class DynamicExprInfo {
    private:

      /// \brief The expression template.
      const char* m_Template;

      std::string m_Result;

      /// \brief The variable list.
      void** m_Addresses;

      /// \brief The variable is set if it is required to print out the result of
      /// the dynamic expression after evaluation
      bool m_ValuePrinterReq;
    public:
      DynamicExprInfo(const char* templ, void* addresses[], bool valuePrinterReq)
        : m_Template(templ), m_Result(templ), m_Addresses(addresses),
          m_ValuePrinterReq(valuePrinterReq) {}

      ///\brief Performs the insertions of the context in the expression just
      /// before evaluation. To be used only at runtime.
      ///
      const char* getExpr();
      bool isValuePrinterRequested() { return m_ValuePrinterReq; }
      const char* getTemplate() const { return m_Template; }
    };
  } // end namespace internal
} // end namespace runtime
} // end namespace cling
#endif // CLING_DYNAMIC_EXPR_INFO_H
