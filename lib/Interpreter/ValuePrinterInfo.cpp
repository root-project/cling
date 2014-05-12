//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/ValuePrinterInfo.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace cling {
  ValuePrinterInfo::ValuePrinterInfo(Interpreter* I, ASTContext* Ctx)
    : m_Type(), m_Interpreter(I), m_Context(Ctx), m_Flags(0) {
    Init(tryGetValuePrintedExpr()->getType());
  }

  ValuePrinterInfo::ValuePrinterInfo(QualType Ty, ASTContext* Ctx)
    : m_Type(), m_Interpreter(0), m_Context(Ctx), m_Flags(0) {
    Init(Ty);
  }

  Expr* ValuePrinterInfo::tryGetValuePrintedExpr() const {
    const Transaction* T = m_Interpreter->getLastTransaction();
    assert(T->getWrapperFD() && "Must have a wrapper.");
    FunctionDecl* FD = T->getWrapperFD();
    Expr* ExprAttachedTo
      = utils::Analyze::GetOrCreateLastExpr(FD, /*foundAtPos*/0, /*omitDS*/false,
                                            &m_Interpreter->getSema());
    return ExprAttachedTo;
  }

  void ValuePrinterInfo::Init(clang::QualType Ty) {
    assert(!Ty.isNull() && "Type must be valid!");
    assert(m_Context && "ASTContext cannot be null!");

    assert(sizeof(m_Type) >= sizeof(clang::QualType) && "m_Type too small!");
    m_Type = *reinterpret_cast<void**>(&Ty);

    // 1. Get the flags
    if (Ty.isLocalConstQualified() || Ty.isConstant(*m_Context)){
      m_Flags |= VPI_Const;
    }

    if (Ty->isPointerType()) {
      // treat arrary-to-pointer decay as array:
      QualType PQT = Ty->getPointeeType();
      const Type* PTT = PQT.getTypePtr();
      if (!PTT || !PTT->isArrayType()) {
        m_Flags |= VPI_Ptr;
        if (const RecordType* RT = dyn_cast<RecordType>(Ty.getTypePtr()))
          if (RecordDecl* RD = RT->getDecl()) {
            CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(RD);
            if (CRD && CRD->isPolymorphic())
              m_Flags |= VPI_Polymorphic;
          }
      }
    }
  }
} // end namespace cling
