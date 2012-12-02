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
  class QualType;
}

namespace cling {

  class ValuePrinterInfo {
  private:
    void* /* clang::QualType */ m_Type; // QualType buffer to prevent #include
    clang::Expr* m_Expr;
    clang::ASTContext* m_Context;
    unsigned m_Flags;

    void Init(clang::QualType Ty);

  public:
    enum ValuePrinterFlags {
      VPI_Ptr = 1,
      VPI_Const = 2,
      VPI_Polymorphic = 4
    };

    ValuePrinterInfo(clang::Expr* Expr, clang::ASTContext* Ctx);
    ValuePrinterInfo(clang::QualType Ty, clang::ASTContext* Ctx);
    const clang::QualType& getType() const {
      return *reinterpret_cast<const clang::QualType*>(&m_Type); }
    clang::Expr* getExpr() const { return m_Expr; }
    clang::ASTContext* getASTContext() const { return m_Context; }
    unsigned getFlags() { return m_Flags; }
  };

} // end namespace cling

#endif // CLING_VALUE_PRINTER_INFO_H
