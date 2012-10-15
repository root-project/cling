//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_SYNTHESIZER_H
#define CLING_VALUE_PRINTER_SYNTHESIZER_H

#include "TransactionTransformer.h"

#include "llvm/ADT/OwningPtr.h"

namespace clang {
  class ASTContext;
  class CompoundStmt;
  class DeclGroupRef;
  class Expr;
  class Sema;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {

  class ValuePrinterSynthesizer : public TransactionTransformer {

  private:
    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    clang::ASTContext* m_Context;

    ///\brief Stream to dump values into.
    ///
    llvm::OwningPtr<llvm::raw_ostream> m_ValuePrinterStream;

public:
    ///\ brief Constructs the value printer synthesizer.
    ///
    ///\param[in] S - The semantic analysis object
    ///\param[in] Stream - The output stream where the value printer will write
    ///                    to. Defaults to std::cout. Owns the stream.
    ValuePrinterSynthesizer(clang::Sema* S, llvm::raw_ostream* Stream);
    
    virtual ~ValuePrinterSynthesizer();

    virtual void Transform();

  private:
    ///\brief Tries to attach a value printing mechanism to the given decl group
    /// ref.
    ///
    ///\param[in] DGR - A decl group ref the value printer is being attached to.
    ///
    ///\returns true if the attachment was considered as success. I.e. even if
    /// even if the value printer wasn't attached because of the compilation 
    /// options disallowint it - it will return still true. Returns false on
    /// critical error.
    bool tryAttachVP(clang::DeclGroupRef DGR);
    clang::Expr* SynthesizeCppVP(clang::Expr* E);
    clang::Expr* SynthesizeVP(clang::Expr* E);
    unsigned ClearNullStmts(clang::CompoundStmt* CS);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
