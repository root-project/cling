//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_SYNTHESIZER_H
#define CLING_VALUE_PRINTER_SYNTHESIZER_H

#include "TransactionTransformer.h"

namespace clang {
  class ASTContext;
  class CompoundStmt;
  class DeclGroupRef;
  class Expr;
  class Sema;
}

namespace cling {
  class Interpreter;

  class ValuePrinterSynthesizer : public TransactionTransformer {

  private:
    Interpreter* m_Interpreter;

    ///\brief Needed for the AST transformations, owned by Sema
    clang::ASTContext* m_Context;

public:
    ValuePrinterSynthesizer(Interpreter* Interp, clang::Sema* S);
    virtual ~ValuePrinterSynthesizer();

    virtual void Transform();

  private:
    bool tryAttachVP(clang::DeclGroupRef DGR);
    clang::Expr* SynthesizeCppVP(clang::Expr* E);
    clang::Expr* SynthesizeVP(clang::Expr* E);
    unsigned ClearNullStmts(clang::CompoundStmt* CS);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
