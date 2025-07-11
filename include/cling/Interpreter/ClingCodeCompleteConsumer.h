//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Bianca-Cristina Cristescu <bianca-cristina.cristescu@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CODE_COMPLETE_CONSUMER
#define CLING_CODE_COMPLETE_CONSUMER

#include "clang/Sema/CodeCompleteConsumer.h"

using namespace clang;

namespace clang {
  class CompilerInstance;
}

namespace cling {
  /// \brief Create a new printing code-completion consumer that prints its
  /// results to the given raw output stream.
  class ClingCodeCompleteConsumer : public CodeCompleteConsumer {
    CodeCompletionTUInfo m_CCTUInfo;
    /// \ brief Results of the completer to be printed by the text interface.
    std::vector<std::string> &m_Completions;

  public:
    ClingCodeCompleteConsumer(const CodeCompleteOptions &CodeComplOpts,
                              std::vector<std::string> &completions)
      : CodeCompleteConsumer(CodeComplOpts),
        m_CCTUInfo(std::make_shared<GlobalCodeCompletionAllocator>()),
        m_Completions(completions) {}
    ~ClingCodeCompleteConsumer() {}

    /// \brief Prints the finalized code-completion results.
    void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                    CodeCompletionResult *Results,
                                    unsigned NumResults) override;

    CodeCompletionAllocator &getAllocator() override {
      return m_CCTUInfo.getAllocator();
    }

    CodeCompletionTUInfo &getCodeCompletionTUInfo() override {
      return m_CCTUInfo;
    }

    bool isResultFilteredOut(StringRef Filter,
                             CodeCompletionResult Results) override;

    void getCompletions(std::vector<std::string>& completions) {
      completions = m_Completions;
    }
  };

  struct ClingCodeCompleter {
    ClingCodeCompleter() = default;
    std::string Prefix;

    /// \param[in] InterpCI The compiler instance that is used to trigger code
    ///                     completion.
    /// \param[in] Content The string where code completion is triggered.
    /// \param[in] Line The line number of the code completion point.
    /// \param[in] Col The column number of the code completion point.
    /// \param[in] ParentCI The running interpreter compiler instance that
    ///                     provides ASTContexts.
    /// \param[out] CCResults The completion results.
    void codeComplete(CompilerInstance* InterpCI, llvm::StringRef Content,
                      unsigned Line, unsigned Col, CompilerInstance* ParentCI,
                      std::vector<std::string>& CCResults);
  };
}

#endif
