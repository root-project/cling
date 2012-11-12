//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_INFO_H
#define CLING_VALUE_PRINTER_INFO_H

#include "clang/AST/Type.h"

namespace clang {
  class ASTContext;
  class Expr;
}

namespace cling {

  class ValuePrinterInfo {
  private:
    clang::QualType m_Type;
    clang::ASTContext* m_Context;
    unsigned m_Flags;

    void Init();

  public:
    enum ValuePrinterFlags {
      VPI_Ptr = 1,
      VPI_Const = 2,
      VPI_Polymorphic = 4
    };

    ValuePrinterInfo(clang::Expr* Expr, clang::ASTContext* Ctx);
    ValuePrinterInfo(clang::QualType Ty, clang::ASTContext* Ctx);
    const clang::QualType getType() const { return m_Type; }
    clang::ASTContext* getASTContext() const { return m_Context; }
    unsigned getFlags() { return m_Flags; }
  };

} // end namespace cling

#endif // CLING_VALUE_PRINTER_INFO_H
