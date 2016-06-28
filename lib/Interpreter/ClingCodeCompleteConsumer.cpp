#include "cling/Interpreter/ClingCodeCompleteConsumer.h"

#include "clang/Sema/Sema.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"


void 
ClingCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &SemaRef,
                                                 CodeCompletionContext Context,
                                                 CodeCompletionResult *Results,
                                                         unsigned NumResults) {
  std::stable_sort(Results, Results + NumResults);
  
  StringRef Filter = SemaRef.getPreprocessor().getCodeCompletionFilter();

  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    if(!Filter.empty() && isResultFilteredOut(Filter, Results[I]))
      continue;
    switch (Results[I].Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef, Context,
                                                    getAllocator(),
                                                    CCTUInfo,
                                                    includeBriefComments())) {
        m_completions.push_back(CCS->getAsString());
      }
      break;
      
    case CodeCompletionResult::RK_Keyword:
      m_completions.push_back(Results[I].Keyword);
      break;
        
    case CodeCompletionResult::RK_Macro: {
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef, Context,
                                                    getAllocator(),
                                                    CCTUInfo,
                                                    includeBriefComments())) {
        m_completions.push_back(CCS->getAsString());
      }
      break;
    }
        
    case CodeCompletionResult::RK_Pattern: {
      m_completions.push_back(Results[I].Pattern->getAsString());
      break;
    }
    }
  }
}

bool ClingCodeCompleteConsumer::isResultFilteredOut(StringRef Filter,
                                                CodeCompletionResult Result) {
  switch (Result.Kind) {
  case CodeCompletionResult::RK_Declaration: {
    return !(Result.Declaration->getIdentifier() &&
            (*Result.Declaration).getIdentifier()->getName().startswith(Filter));
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
