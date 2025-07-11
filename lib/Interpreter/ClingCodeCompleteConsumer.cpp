//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Bianca-Cristina Cristescu <bianca-cristina.cristescu@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/ClingCodeCompleteConsumer.h"

#include "clang/AST/ASTImporter.h"
#include "clang/AST/DeclLookups.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
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
        return !(
            Result.Declaration->getIdentifier() &&
            Result.Declaration->getIdentifier()->getName().starts_with(Filter));
      }
      case CodeCompletionResult::RK_Keyword: {
        return !((StringRef(Result.Keyword)).starts_with(Filter));
      }
      case CodeCompletionResult::RK_Macro: {
        return !(Result.Macro->getName().starts_with(Filter));
      }
      case CodeCompletionResult::RK_Pattern: {
        return !(
            StringRef((Result.Pattern->getAsString())).starts_with(Filter));
      }
      default: llvm_unreachable("Unknown code completion result Kind.");
    }
  }

  // Code copied from: clang/lib/Interpreter/CodeCompletion.cpp
  class IncrementalSyntaxOnlyAction : public clang::SyntaxOnlyAction {
    const CompilerInstance* ParentCI;

  public:
    IncrementalSyntaxOnlyAction(const CompilerInstance* ParentCI)
        : ParentCI(ParentCI) {}

  protected:
    void ExecuteAction() override;
  };

  class ExternalSource : public clang::ExternalASTSource {
    TranslationUnitDecl* ChildTUDeclCtxt;
    ASTContext& ParentASTCtxt;
    TranslationUnitDecl* ParentTUDeclCtxt;

    std::unique_ptr<ASTImporter> Importer;

  public:
    ExternalSource(ASTContext& ChildASTCtxt, FileManager& ChildFM,
                   ASTContext& ParentASTCtxt, FileManager& ParentFM);
    bool FindExternalVisibleDeclsByName(const DeclContext* DC,
                                        DeclarationName Name) override;
    void completeVisibleDeclsMap(
        const clang::DeclContext* childDeclContext) override;
  };

  // This method is intended to set up `ExternalASTSource` to the running
  // compiler instance before the super `ExecuteAction` triggers parsing
  void IncrementalSyntaxOnlyAction::ExecuteAction() {
    clang::CompilerInstance& CI = getCompilerInstance();
    ExternalSource* myExternalSource =
        new ExternalSource(CI.getASTContext(), CI.getFileManager(),
                           ParentCI->getASTContext(),
                           ParentCI->getFileManager());
    llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> astContextExternalSource(
        myExternalSource);
    CI.getASTContext().setExternalSource(astContextExternalSource);
    CI.getASTContext().getTranslationUnitDecl()->setHasExternalVisibleStorage(
        true);

    // Load all external decls into current context. Under the hood, it calls
    // ExternalSource::completeVisibleDeclsMap, which make all decls on the
    // redecl chain visible.
    //
    // This is crucial to code completion on dot members, since a bound variable
    // before "." would be otherwise treated out-of-scope.
    //
    // clang-repl> Foo f1;
    // clang-repl> f1.<tab>
    CI.getASTContext().getTranslationUnitDecl()->lookups();
    SyntaxOnlyAction::ExecuteAction();
  }

  ExternalSource::ExternalSource(ASTContext& ChildASTCtxt, FileManager& ChildFM,
                                 ASTContext& ParentASTCtxt,
                                 FileManager& ParentFM)
      : ChildTUDeclCtxt(ChildASTCtxt.getTranslationUnitDecl()),
        ParentASTCtxt(ParentASTCtxt),
        ParentTUDeclCtxt(ParentASTCtxt.getTranslationUnitDecl()) {
    clang::ASTImporter* importer =
        new clang::ASTImporter(ChildASTCtxt, ChildFM, ParentASTCtxt, ParentFM,
                               /*MinimalImport : ON*/ true);
    Importer.reset(importer);
  }

  bool ExternalSource::FindExternalVisibleDeclsByName(const DeclContext* DC,
                                                      DeclarationName Name) {

    IdentifierTable& ParentIdTable = ParentASTCtxt.Idents;

    auto ParentDeclName =
        DeclarationName(&(ParentIdTable.get(Name.getAsString())));

    DeclContext::lookup_result lookup_result =
        ParentTUDeclCtxt->lookup(ParentDeclName);

    if (!lookup_result.empty()) {
      return true;
    }
    return false;
  }

  void
  ExternalSource::completeVisibleDeclsMap(const DeclContext* ChildDeclContext) {
    assert(ChildDeclContext && ChildDeclContext == ChildTUDeclCtxt &&
           "No child decl context!");

    if (!ChildDeclContext->hasExternalVisibleStorage())
      return;

    for (auto* DeclCtxt = ParentTUDeclCtxt; DeclCtxt != nullptr;
         DeclCtxt = DeclCtxt->getPreviousDecl()) {
      for (auto& IDeclContext : DeclCtxt->decls()) {
        if (!llvm::isa<NamedDecl>(IDeclContext))
          continue;

        NamedDecl* Decl = llvm::cast<NamedDecl>(IDeclContext);

        auto DeclOrErr = Importer->Import(Decl);
        if (!DeclOrErr) {
          // if an error happens, it usually means the decl has already been
          // imported or the decl is a result of a failed import.  But in our
          // case, every import is fresh each time code completion is
          // triggered. So Import usually doesn't fail. If it does, it just
          // means the related decl can't be used in code completion and we can
          // safely drop it.
          llvm::consumeError(DeclOrErr.takeError());
          continue;
        }

        if (!llvm::isa<NamedDecl>(*DeclOrErr))
          continue;

        NamedDecl* importedNamedDecl = llvm::cast<NamedDecl>(*DeclOrErr);

        SetExternalVisibleDeclsForName(ChildDeclContext,
                                       importedNamedDecl->getDeclName(),
                                       importedNamedDecl);

        if (!llvm::isa<CXXRecordDecl>(importedNamedDecl))
          continue;

        auto* Record = llvm::cast<CXXRecordDecl>(importedNamedDecl);

        if (auto Err = Importer->ImportDefinition(Decl)) {
          // the same as above
          llvm::consumeError(std::move(Err));
          continue;
        }

        Record->setHasLoadedFieldsFromExternalStorage(true);
        for (auto* Meth : Record->methods())
          SetExternalVisibleDeclsForName(ChildDeclContext, Meth->getDeclName(),
                                         Meth);
      }
      ChildDeclContext->setHasExternalLexicalStorage(false);
    }
  }

  // Copied and adapted from: clang/lib/Interpreter/CodeCompletion.cpp
  void ClingCodeCompleter::codeComplete(clang::CompilerInstance* InterpCI,
                                        llvm::StringRef Content, unsigned Line,
                                        unsigned Col,
                                        clang::CompilerInstance* ParentCI,
                                        std::vector<std::string>& CCResults) {
    const std::string CodeCompletionFileName = "input_line_[Completion]";
    auto DiagOpts = DiagnosticOptions();
    auto consumer =
        ClingCodeCompleteConsumer(ParentCI->getFrontendOpts().CodeCompleteOpts,
                                  CCResults);
    ;

    auto diag = InterpCI->getDiagnosticsPtr();
    std::unique_ptr<clang::ASTUnit> AU(
        clang::ASTUnit::LoadFromCompilerInvocationAction(
            InterpCI->getInvocationPtr(),
            std::make_shared<PCHContainerOperations>(), diag));
    llvm::SmallVector<clang::StoredDiagnostic, 8> sd = {};
    llvm::SmallVector<const llvm::MemoryBuffer*, 1> tb = {};
    InterpCI->getFrontendOpts().Inputs[0] =
        FrontendInputFile(CodeCompletionFileName, Language::CXX,
                          InputKind::Source);
    auto Act = std::unique_ptr<IncrementalSyntaxOnlyAction>(
        new IncrementalSyntaxOnlyAction(ParentCI));
    std::unique_ptr<llvm::MemoryBuffer> MB =
        llvm::MemoryBuffer::getMemBufferCopy(Content, CodeCompletionFileName);
    llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;

    RemappedFiles.push_back(std::make_pair(CodeCompletionFileName, MB.get()));
    // we don't want the AU destructor to release the memory buffer that MB
    // owns twice, because MB handles its resource on its own.
    AU->setOwnsRemappedFileBuffers(false);
    AU->CodeComplete(CodeCompletionFileName, 1, Col, RemappedFiles, false,
                     false, false, consumer,
                     std::make_shared<clang::PCHContainerOperations>(), *diag,
                     InterpCI->getLangOpts(), InterpCI->getSourceManager(),
                     InterpCI->getFileManager(), sd, tb, std::move(Act));
  }
}
