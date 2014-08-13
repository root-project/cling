//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Manasij Mukherjee  <manasij7479@gmail.com>
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

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

      if (m_Map->find(FE) == m_Map->end())
        (*m_Map)[FE] = std::vector<Decl*>();

      (*m_Map)[FE].push_back(decl);
    }

  public:
    DefaultArgVisitor() : m_IsStoringState(false), m_Map(0) {}
    void RemoveDefaultArgsOf(Decl* D) {
      //D = D->getMostRecentDecl();
      TraverseDecl(D);
      //while ((D = D->getPreviousDecl()))
      //  TraverseDecl(D);
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

    bool shouldVisitTemplateInstantiations() { return true; }
    bool TraverseTemplateTypeParmDecl(TemplateTypeParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool VisitDecl(Decl* D) {
      if (!m_IsStoringState)
        return true;

      if (!D->hasAttr<AnnotateAttr>())
        return false;

      AnnotateAttr* attr = D->getAttr<AnnotateAttr>();
      if (!attr)
        return true;

      switch (D->getKind()) {
      default:
        InsertIntoAutoloadingState(D, attr->getAnnotation());
        break;
      case Decl::Enum:
        // EnumDecls have extra information 2 chars after the filename used
        // for extra fixups.
        InsertIntoAutoloadingState(D, attr->getAnnotation().drop_back(2));
        break;
      }

      return true;
    }

    bool TraverseTemplateDecl(TemplateDecl* D) {
      if (!D->getTemplatedDecl()->hasAttr<AnnotateAttr>())
        return true;
      for(auto P: D->getTemplateParameters()->asArray())
        TraverseDecl(P);
      return true;
    }

    bool TraverseNonTypeTemplateParmDecl(NonTypeTemplateParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool TraverseParmVarDecl(ParmVarDecl* D) {
      if (m_IsStoringState)
        return true;

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

    DefaultArgVisitor defaultArgsCleaner;
    for (auto D : found->second) {
      defaultArgsCleaner.RemoveDefaultArgsOf(D);
    }
    // Don't need to keep track of cleaned up decls from file.
    m_Map.erase(found);
  }

  AutoloadCallback::AutoloadCallback(Interpreter* interp) :
    InterpreterCallbacks(interp,true,false,true), m_Interpreter(interp){
//#ifdef _POSIX_C_SOURCE
//      //Workaround for differnt expansion of macros to typedefs
//      m_Interpreter->parse("#include <sys/types.h>");
//#endif
  }

  AutoloadCallback::~AutoloadCallback() {
  }

  void AutoloadCallback::TransactionCommitted(const Transaction &T) {
    if (T.decls_begin() == T.decls_end())
      return;
    if (T.decls_begin()->m_DGR.isNull())
      return;

    if (const NamedDecl* ND = dyn_cast<NamedDecl>(*T.decls_begin()->m_DGR.begin()))
      if (ND->getIdentifier() && ND->getName().equals("__Cling_Autoloading_Map")) {
        DefaultArgVisitor defaultArgsStateCollector;
        Preprocessor& PP = m_Interpreter->getCI()->getPreprocessor();
        for (Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
             I != E; ++I) {
          Transaction::DelayCallInfo DCI = *I;

          // if (DCI.m_Call != Transaction::kCCIHandleTopLevelDecl)
          //   continue;
          if (DCI.m_DGR.isNull())
            continue;

          for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
                 JE = DCI.m_DGR.end(); J != JE; ++J) {
            defaultArgsStateCollector.TrackDefaultArgStateOf(*J, m_Map, PP);
          }
        }
      }
  }

} //end namespace cling
