#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/AutoloadCallback.h"
#include "cling/Interpreter/Transaction.h"


using namespace clang;

namespace cling {

  void AutoloadCallback::report(clang::SourceLocation l,std::string name,std::string header) {
    Sema& sema= m_Interpreter->getSema();

    unsigned id
      = sema.getDiagnostics().getCustomDiagID (DiagnosticsEngine::Level::Warning,
                                                 "Note: '%0' can be found in %1");
/*    unsigned idn //TODO: To be enabled after we have a way to get the full path
      = sema.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Level::Note,
                                                "Type : %0 , Full Path: %1")*/;

    sema.Diags.Report(l, id) << name << header;

  }

  bool AutoloadCallback::LookupObject (TagDecl *t) {
    if (t->hasAttr<AnnotateAttr>())
      report(t->getLocation(),t->getNameAsString(),t->getAttr<AnnotateAttr>()->getAnnotation());
    return false;
  }

  class DefaultArgVisitor: public RecursiveASTVisitor<DefaultArgVisitor> {
  private:
    bool m_IsStoringState;
    AutoloadCallback::FwdDeclsMap* m_Map;
    clang::Preprocessor* m_PP;
  private:
    void InsertIntoAutoloadingState (Decl* decl, std::string annotation) {

      assert(annotation != "" && "Empty annotation!");
      assert(m_PP);

      const FileEntry* FE = 0;
      SourceLocation fileNameLoc;
      bool isAngled = false;
      const DirectoryLookup* LookupFrom = 0;
      const DirectoryLookup* CurDir = 0;

      FE = m_PP->LookupFile(fileNameLoc, annotation, isAngled, LookupFrom,
                            CurDir, /*SearchPath*/0, /*RelativePath*/ 0,
                            /*suggestedModule*/0, /*SkipCache*/false,
                            /*OpenFile*/ false, /*CacheFail*/ false);

      assert(FE && "Must have a valid FileEntry");

      if(m_Map->find(FE) == m_Map->end())
        (*m_Map)[FE] = std::vector<Decl*>();

      (*m_Map)[FE].push_back(decl);
    }

  public:
    DefaultArgVisitor() : m_IsStoringState(false), m_Map(0) {}

    void RemoveDefaultArgsOf(Decl* D) {
      TraverseDecl(D);
    }

    void TrackDefaultArgStateOf(Decl* D, AutoloadCallback::FwdDeclsMap& map,
                                Preprocessor& PP) {
      m_IsStoringState = true;
      m_Map = &map;
      m_PP = &PP;
      TraverseDecl(D);
      m_PP = 0;
      m_Map = 0;
      m_IsStoringState = false;
    }

    bool VisitTemplateTypeParmDecl(TemplateTypeParmDecl* D) {
      if (m_IsStoringState) {
        Decl* parent = cast<Decl>(D->getDeclContext());
        if (AnnotateAttr* attr = parent->getAttr<AnnotateAttr>())
          InsertIntoAutoloadingState(D, attr->getAnnotation());
      }
      else if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl* D) {
      if (m_IsStoringState) {
        Decl* parent = cast<Decl>(D->getDeclContext());
        if (AnnotateAttr* attr = parent->getAttr<AnnotateAttr>())
          InsertIntoAutoloadingState(D, attr->getAnnotation());
      }
      if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool VisitParmVarDecl(ParmVarDecl* D) {
      if (m_IsStoringState) {
        Decl* parent = cast<Decl>(D->getDeclContext());
        if (AnnotateAttr* attr = parent->getAttr<AnnotateAttr>())
          InsertIntoAutoloadingState(D, attr->getAnnotation());
      }
      if (D->hasDefaultArg())
        D->setDefaultArg(nullptr);
      return true;
    }
  };

  void AutoloadCallback::InclusionDirective(clang::SourceLocation HashLoc,
                          const clang::Token &IncludeTok,
                          llvm::StringRef FileName,
                          bool IsAngled,
                          clang::CharSourceRange FilenameRange,
                          const clang::FileEntry *File,
                          llvm::StringRef SearchPath,
                          llvm::StringRef RelativePath,
                          const clang::Module *Imported) {
    assert(File && "Must have a valid File");

    auto found = m_Map.find(File);
    if (found == m_Map.end())
      return; // nothing to do, file not referred in any annotation
    // if(iterator->second.Included)
    //   return; // nothing to do, file already included once

    DefaultArgVisitor defaultArgsCleaner;

    for (auto D : found->second) {
      D->dropAttrs();
      defaultArgsCleaner.RemoveDefaultArgsOf(D);
    }
    // Don't need to keep track of cleaned up decls from file.
    m_Map.erase(found);
  }

  AutoloadCallback::AutoloadCallback(Interpreter* interp) :
    InterpreterCallbacks(interp,true,false,true), m_Interpreter(interp){

  }

  AutoloadCallback::~AutoloadCallback() {
  }

  void AutoloadCallback::TransactionCommitted(const Transaction &T) {

    DefaultArgVisitor defaultArgsStateCollector;

    for (Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
         I != E; ++I) {
      Transaction::DelayCallInfo DCI = *I;

      if (DCI.m_Call != Transaction::kCCIHandleTopLevelDecl)
        continue;
      if (DCI.m_DGR.isNull() || (*DCI.m_DGR.begin())->hasAttr<AnnotateAttr>())
        continue;

      for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
             JE = DCI.m_DGR.end(); J != JE; ++J) {
        defaultArgsStateCollector.TrackDefaultArgStateOf(*J, m_Map,
                                                         m_Interpreter->getCI()->getPreprocessor());
      }
    }
  }

} //end namespace cling
