//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_SYNTHESIZER_H
#define CLING_VALUE_PRINTER_SYNTHESIZER_H

#include "ASTTransformer.h"

#include <memory>

namespace clang {
  class ASTContext;
  class CompoundStmt;
  class Decl;
  class FunctionDecl;
  class Expr;
  class Sema;
  class LookupResult;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {

  class ValuePrinterSynthesizer : public WrapperTransformer {

  private:
    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    clang::ASTContext* m_Context;

    ///\brief cling runtime "Cannot find cling_PrintValue(...)" cache.
    ///
    clang::LookupResult* m_LookupResult;

public:
    ///\ brief Constructs the value printer synthesizer.
    ///
    ///\param[in] S - The semantic analysis object
    ValuePrinterSynthesizer(clang::Sema* S);

    virtual ~ValuePrinterSynthesizer();

    Result Transform(clang::Decl* D) override;

  private:
    ///\brief Tries to attach a value printing mechanism to the given decl group
    /// ref.
    ///
    ///\param[in] FD - wrapper function that the value printer will attached to.
    ///
    ///\returns true if the attachment was considered as success. I.e. even if
    /// even if the value printer wasn't attached because of the compilation
    /// options disallowint it - it will return still true. Returns false on
    /// critical error.
    bool tryAttachVP(clang::FunctionDecl* FD);
    clang::Expr* SynthesizeVP(clang::Expr* E);
    unsigned ClearNullStmts(clang::CompoundStmt* CS);

    // Find and cache cling::runtime on first request.
    void FindAndCacheRuntimeLookupResult(clang::SourceLocation SourceLoc);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
