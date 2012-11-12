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
  ValuePrinterInfo::ValuePrinterInfo(Expr* E, ASTContext* Ctx)
    : m_Type(E->getType()), m_Context(Ctx), m_Flags(0) {
    Init();
  }

  ValuePrinterInfo::ValuePrinterInfo(QualType Ty, ASTContext* Ctx)
    : m_Type(Ty), m_Context(Ctx), m_Flags(0) {
    Init();
  }

  void ValuePrinterInfo::Init() {
    assert(!m_Type.isNull() && "Type must be valid!");
    assert(m_Context && "ASTContext cannot be null!");
    // 1. Get the flags

    if (m_Type.isLocalConstQualified() || m_Type.isConstant(*m_Context)){
      m_Flags |= VPI_Const;
    }

    if (m_Type->isPointerType()) {
      // treat arrary-to-pointer decay as array:
      QualType PQT = m_Type->getPointeeType();
      const Type* PTT = PQT.getTypePtr();
      if (!PTT || !PTT->isArrayType()) {
        m_Flags |= VPI_Ptr;
        if (const RecordType* RT = dyn_cast<RecordType>(m_Type.getTypePtr()))
          if (RecordDecl* RD = RT->getDecl()) {
            CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(RD);
            if (CRD && CRD->isPolymorphic())
              m_Flags |= VPI_Polymorphic;
          }
      }
    }
  }
} // end namespace cling
