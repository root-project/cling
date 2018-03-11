//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_EXTRACTION_SYNTHESIZER_H
#define CLING_VALUE_EXTRACTION_SYNTHESIZER_H

#include "ASTTransformer.h"

namespace clang {
  class ASTContext;
  class Decl;
  class Expr;
  class Sema;
  class VarDecl;
}

namespace cling {

  class ValueExtractionSynthesizer : public WrapperTransformer {

  private:
    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    clang::ASTContext* m_Context;

    ///\brief cling::runtime::gCling variable cache.
    ///
    clang::VarDecl* m_gClingVD;

    ///\brief cling::runtime::internal::setValueNoAlloc cache.
    ///
    clang::Expr* m_UnresolvedNoAlloc;

    ///\brief cling::runtime::internal::setValueWithAlloc cache.
    ///
    clang::Expr* m_UnresolvedWithAlloc;

    ///\brief cling::runtime::internal::copyArray cache.
    ///
    clang::Expr* m_UnresolvedCopyArray;

    bool m_isChildInterpreter;

public:
    ///\ brief Constructs the return synthesizer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///\param[in] isChildInterpreter - flag to control if it is called
    /// from a child or parent Interpreter
    ///
    ValueExtractionSynthesizer(clang::Sema* S, bool isChildInterpreter);

    virtual ~ValueExtractionSynthesizer();

    Result Transform(clang::Decl* D) override;

  private:

    ///\brief
    /// Here we don't want to depend on the JIT runFunction, because of its
    /// limitations, when it comes to return value handling. There it is
    /// not clear who provides the storage and who cleans it up in a
    /// platform independent way.
    //
    /// Depending on the type we need to synthesize a call to cling:
    /// 0) void : do nothing;
    /// 1) enum, integral, float, double, referece, pointer types :
    ///      call to cling::internal::setValueNoAlloc(...);
    /// 2) object type (alloc on the stack) :
    ///      cling::internal::setValueWithAlloc
    ///   2.1) constant arrays:
    ///          call to cling::runtime::internal::copyArray(...)
    ///
    /// We need to synthesize later:
    /// Wrapper has signature: void w(cling::Value V)
    /// case 1):
    ///   setValueNoAlloc(gCling, &SVR, lastExprTy, lastExpr())
    /// case 2):
    ///   new (setValueWithAlloc(gCling, &SVR, lastExprTy)) (lastExpr)
    /// case 2.1):
    ///   copyArray(src, placement, N)
    ///
    clang::Expr* SynthesizeSVRInit(clang::Expr* E);

    // Find and cache cling::runtime::gCling, setValueNoAlloc,
    // setValueWithAlloc on first request.
    void FindAndCacheRuntimeDecls();
  };

} // namespace cling

#endif // CLING_VALUE_EXTRACTION_SYNTHESIZER_H
