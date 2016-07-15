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
      OS << *Results[I].Declaration;
      if (Results[I].Hidden)
        OS << " (Hidden)";
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef, Context,
                                                    getAllocator(),
                                                    CCTUInfo,
                                                    includeBriefComments())) {
        OS << " : " << CCS->getAsString();
        if (const char *BriefComment = CCS->getBriefComment())
          OS << " : " << BriefComment;
      }
        
      OS << '\n';
      break;
      
    case CodeCompletionResult::RK_Keyword:
      OS << Results[I].Keyword << '\n';
      break;
        
    case CodeCompletionResult::RK_Macro: {
      OS << Results[I].Macro->getName();
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef, Context,
                                                    getAllocator(),
                                                    CCTUInfo,
                                                    includeBriefComments())) {
        OS << " : " << CCS->getAsString();
      }
      OS << '\n';
      break;
    }
        
    case CodeCompletionResult::RK_Pattern: {
      OS << "Pattern : " 
         << Results[I].Pattern->getAsString() << '\n';
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
