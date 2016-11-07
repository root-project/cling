//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/ParserStateRAII.h"

#include "clang/Parse/RAIIObjectsForParser.h"

using namespace clang;

cling::ParserStateRAII::ParserStateRAII(Parser& p)
  : P(&p), PP(p.getPreprocessor()),
    ResetIncrementalProcessing(p.getPreprocessor()
                               .isIncrementalProcessingEnabled()),
    OldSuppressAllDiagnostics(P->getActions().getDiagnostics()
                              .getSuppressAllDiagnostics()),
    OldPPSuppressAllDiagnostics(p.getPreprocessor().getDiagnostics()
                                .getSuppressAllDiagnostics()),
    OldSpellChecking(p.getPreprocessor().getLangOpts().SpellChecking),
  OldPrevTokLocation(p.PrevTokLocation),
  OldParenCount(p.ParenCount), OldBracketCount(p.BracketCount),
  OldBraceCount(p.BraceCount),
  OldTemplateParameterDepth(p.TemplateParameterDepth),
  OldInNonInstantiationSFINAEContext(P->getActions()
                                     .InNonInstantiationSFINAEContext)
{
  OldTemplateIds.swap(P->TemplateIds);
}

cling::ParserStateRAII::~ParserStateRAII() {
  //
  // Advance the parser to the end of the file, and pop the include stack.
  //
  // Note: Consuming the EOF token will pop the include stack.
  //
  {
    // Cleanup the TemplateIds before swapping the previous set back.
    DestroyTemplateIdAnnotationsRAIIObj CleanupTemplateIds(*P);
  }
  P->TemplateIds.swap(OldTemplateIds);
  P->SkipUntil(tok::eof);
  PP.enableIncrementalProcessing(ResetIncrementalProcessing);
  // Doesn't reset the diagnostic mappings
  P->getActions().getDiagnostics().Reset(/*soft=*/true);
  P->getActions().getDiagnostics().setSuppressAllDiagnostics(OldSuppressAllDiagnostics);
  PP.getDiagnostics().Reset(/*soft=*/true);
  PP.getDiagnostics().setSuppressAllDiagnostics(OldPPSuppressAllDiagnostics);
  const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking =
    OldSpellChecking;

  P->PrevTokLocation = OldPrevTokLocation;
  P->ParenCount = OldParenCount;
  P->BracketCount = OldBracketCount;
  P->BraceCount = OldBraceCount;
  P->TemplateParameterDepth = OldTemplateParameterDepth;
  P->getActions().InNonInstantiationSFINAEContext =
    OldInNonInstantiationSFINAEContext;
}
