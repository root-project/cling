//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Bianca-Cristina Cristescu <bianca-cristina.cristescu@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/ClingCodeCompleteConsumer.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

namespace cling {
  void ClingCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &SemaRef,
                                                   CodeCompletionContext Context,
                                                   CodeCompletionResult *Results,
                                                           unsigned NumResults) {
    std::stable_sort(Results, Results + NumResults);

    StringRef Filter = SemaRef.getPreprocessor().getCodeCompletionFilter();

    for (unsigned I = 0; I != NumResults; ++I) {
      if (!Filter.empty() && isResultFilteredOut(Filter, Results[I]))
        continue;
      switch (Results[I].Kind) {
        case CodeCompletionResult::RK_Declaration:
          if (CodeCompletionString *CCS
              = Results[I].CreateCodeCompletionString(SemaRef, Context,
                                                      getAllocator(),
                                                      m_CCTUInfo,
                                                      includeBriefComments())) {
            m_Completions.push_back(CCS->getAsString());
          }
          break;

        case CodeCompletionResult::RK_Keyword:
          m_Completions.push_back(Results[I].Keyword);
          break;

        case CodeCompletionResult::RK_Macro:
          if (CodeCompletionString *CCS
              = Results[I].CreateCodeCompletionString(SemaRef, Context,
                                                      getAllocator(),
                                                      m_CCTUInfo,
                                                      includeBriefComments())) {
            m_Completions.push_back(CCS->getAsString());
          }
          break;

        case CodeCompletionResult::RK_Pattern:
          m_Completions.push_back(Results[I].Pattern->getAsString());
          break;
      }
    }
  }

  bool ClingCodeCompleteConsumer::isResultFilteredOut(StringRef Filter,
                                                  CodeCompletionResult Result) {
    switch (Result.Kind) {
      case CodeCompletionResult::RK_Declaration: {
        return !(Result.Declaration->getIdentifier() &&
            Result.Declaration->getIdentifier()->getName().startswith(Filter));
      }
      case CodeCompletionResult::RK_Keyword: {
        return !((StringRef(Result.Keyword)).startswith(Filter));
      }
      case CodeCompletionResult::RK_Macro: {
        return !(Result.Macro->getName().startswith(Filter));
      }
      case CodeCompletionResult::RK_Pattern: {
        return !(StringRef((Result.Pattern->getAsString())).startswith(Filter));
      }
      default: llvm_unreachable("Unknown code completion result Kind.");
    }
  }
}
