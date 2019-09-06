//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_ParserStateRAII_H
#define CLING_UTILS_ParserStateRAII_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Parse/Parser.h"

namespace clang {
  class Preprocessor;
}

namespace cling {
  ///\brief Cleanup Parser state after a failed lookup.
  ///
  /// After a failed lookup we need to discard the remaining unparsed input,
  /// restore the original state of the incremental parsing flag, clear any
  /// pending diagnostics, restore the suppress diagnostics flag, and restore
  /// the spell checking language options.
  ///
  class ParserStateRAII {
  private:
    clang::Parser* P;
    clang::Preprocessor& PP;
    decltype(clang::Parser::TemplateIds) OldTemplateIds;
    bool ResetIncrementalProcessing;
    bool PPDiagHadErrors;
    bool SemaDiagHadErrors;
    bool OldSuppressAllDiagnostics;
    bool OldPPSuppressAllDiagnostics;
    bool OldSpellChecking;
    clang::Token OldTok;
    clang::SourceLocation OldPrevTokLocation;
    unsigned short OldParenCount, OldBracketCount, OldBraceCount;
    unsigned OldTemplateParameterDepth;
    bool OldInNonInstantiationSFINAEContext;
    bool SkipToEOF;

  public:
    ParserStateRAII(clang::Parser& p, bool skipToEOF);
    ~ParserStateRAII();

    void SetSkipToEOF(bool newvalue) { SkipToEOF = newvalue; }
};

} // end namespace cling
#endif // CLING_UTILS_ParserStateRAII_H
