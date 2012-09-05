//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_INFO_H
#define CLING_VALUE_PRINTER_INFO_H

namespace clang {
  class ASTContext;
  class Expr;
}

namespace cling {

  class ValuePrinterInfo {
  private:
    const clang::Expr* m_Expr;
    const clang::ASTContext* m_Context;
    unsigned m_Flags;

  public:
    enum ValuePrinterFlags {
      VPI_Ptr = 1,
      VPI_Const = 2,
      VPI_Polymorphic = 4
    };

    ValuePrinterInfo(clang::Expr* E, clang::ASTContext* Ctx);
    const clang::Expr* getExpr() const { return m_Expr; }
    const clang::ASTContext* getASTContext() const { return m_Context; }
    unsigned getFlags() { return m_Flags; }
  };

} // end namespace cling

#endif // CLING_VALUE_PRINTER_INFO_H
