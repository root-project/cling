//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/ValuePrinterInfo.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace cling {
  ValuePrinterInfo::ValuePrinterInfo(const Expr* E, ASTContext* Ctx)
    : m_Expr(E), m_Context(Ctx), m_Flags(0) {
    assert(E && "Expression cannot be null!");
    assert(Ctx && "ASTContext cannot be null!");
    // 1. Get the flags
    const QualType QT = m_Expr->getType();

    if (E->isRValue() || QT.isLocalConstQualified() || QT.isConstant(*Ctx)){
      m_Flags |= VPI_Const;
    }

    if (QT->isPointerType()) {
      // treat arrary-to-pointer decay as array:
      QualType PQT = QT->getPointeeType();
      const Type* PTT = PQT.getTypePtr();
      if (!PTT || !PTT->isArrayType()) {
        m_Flags |= VPI_Ptr;
        if (const RecordType* RT = dyn_cast<RecordType>(QT.getTypePtr()))
          if (RecordDecl* RD = RT->getDecl()) {
            CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(RD);
            if (CRD && CRD->isPolymorphic())
              m_Flags |= VPI_Polymorphic;
          }
      }
    }
  }
} // end namespace cling
