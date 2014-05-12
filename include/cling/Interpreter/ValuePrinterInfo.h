//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_INFO_H
#define CLING_VALUE_PRINTER_INFO_H

namespace clang {
  class ASTContext;
  class Expr;
  class QualType;
}

namespace cling {
  class Interpreter;

  class ValuePrinterInfo {
  private:
    void* /* clang::QualType */ m_Type; // QualType buffer to prevent #include
    Interpreter* m_Interpreter;
    clang::ASTContext* m_Context;
    unsigned m_Flags;

    void Init(clang::QualType Ty);

  public:
    enum ValuePrinterFlags {
      VPI_Ptr = 1,
      VPI_Const = 2,
      VPI_Polymorphic = 4
    };

    ValuePrinterInfo(Interpreter* I, clang::ASTContext* Ctx);
    ValuePrinterInfo(clang::QualType Ty, clang::ASTContext* Ctx);
    clang::Expr* tryGetValuePrintedExpr() const;
    const clang::QualType& getType() const {
      return *reinterpret_cast<const clang::QualType*>(&m_Type); }
    clang::ASTContext* getASTContext() const { return m_Context; }
    unsigned getFlags() { return m_Flags; }
  };

} // end namespace cling

#endif // CLING_VALUE_PRINTER_INFO_H
