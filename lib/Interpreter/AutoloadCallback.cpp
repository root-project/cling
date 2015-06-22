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
#include "clang/AST/ASTContext.h" // for operator new[](unsigned long, ASTCtx..)
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/AutoloadCallback.h"
#include "cling/Interpreter/Transaction.h"
#include "TransactionUnloader.h"

namespace {
  static const char annoTag[] = "$clingAutoload$";
  static const size_t lenAnnoTag = sizeof(annoTag) - 1;
}

using namespace clang;

namespace cling {

  void AutoloadCallback::report(clang::SourceLocation l, llvm::StringRef name,
                                llvm::StringRef header) {
    Sema& sema= m_Interpreter->getSema();

    unsigned id
      = sema.getDiagnostics().getCustomDiagID (DiagnosticsEngine::Level::Warning,
                                                 "Note: '%0' can be found in %1");
/*    unsigned idn //TODO: To be enabled after we have a way to get the full path
      = sema.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Level::Note,
                                                "Type : %0 , Full Path: %1")*/;

    if (header.startswith(llvm::StringRef(annoTag, lenAnnoTag)))
      sema.Diags.Report(l, id) << name << header.drop_front(lenAnnoTag);

  }

  bool AutoloadCallback::LookupObject (TagDecl *t) {
    if (m_ShowSuggestions && t->hasAttr<AnnotateAttr>())
      report(t->getLocation(),t->getNameAsString(),t->getAttr<AnnotateAttr>()->getAnnotation());
    return false;
  }

  class AutoloadingVisitor: public RecursiveASTVisitor<AutoloadingVisitor> {
  private:
    ///\brief Flag determining the visitor's actions. If true, register autoload
    /// entries, i.e. remember the connection between filename and the declaration
    /// that needs to be updated on #include of the filename.
    /// If false, react on an #include by adjusting the forward decls, e.g. by
    /// removing the default tremplate arguments (that will now be provided by
    /// the definition read from the include) and by removing enum declarations
    /// that would otherwise be duplicates.
    bool m_IsStoringState;
    AutoloadCallback::FwdDeclsMap* m_Map;
    clang::Preprocessor* m_PP;
    clang::Sema* m_Sema;
    const clang::FileEntry* m_PrevFE;
    std::string m_PrevFileName;
  private:
    void InsertIntoAutoloadingState (Decl* decl, llvm::StringRef annotation) {

      assert(!annotation.empty() && "Empty annotation!");
      if (!annotation.startswith(llvm::StringRef(annoTag, lenAnnoTag))) {
        // not an autoload annotation.
        return;
      }

      assert(m_PP);

      const FileEntry* FE = 0;
      SourceLocation fileNameLoc;
      bool isAngled = false;
      const DirectoryLookup* FromDir = 0;
      const FileEntry* FromFile = 0;
      const DirectoryLookup* CurDir = 0;
      llvm::StringRef FileName = annotation.drop_front(lenAnnoTag);
      if (FileName.equals(m_PrevFileName))
        FE = m_PrevFE;
      else {
        FE = m_PP->LookupFile(fileNameLoc, FileName, isAngled,
                              FromDir, FromFile, CurDir, /*SearchPath*/0,
                              /*RelativePath*/ 0, /*suggestedModule*/0,
                              /*SkipCache*/ false, /*OpenFile*/ false,
                              /*CacheFail*/ true);
        m_PrevFE = FE;
        m_PrevFileName = FileName;
      }

      assert(FE && "Must have a valid FileEntry");
      if (FE) {
        auto& Vec = (*m_Map)[FE];
        Vec.push_back(decl);
      }
    }

  public:
    AutoloadingVisitor():
      m_IsStoringState(false), m_Map(0), m_PP(0), m_Sema(0), m_PrevFE(0) {}
    void RemoveDefaultArgsOf(Decl* D, Sema* S) {
      //D = D->getMostRecentDecl();
      m_Sema = S;
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

    bool VisitDecl(Decl* D) {
      if (!m_IsStoringState)
        return true;

      if (!D->hasAttr<AnnotateAttr>())
        return true;

      if (AnnotateAttr* attr = D->getAttr<AnnotateAttr>())
        InsertIntoAutoloadingState(D, attr->getAnnotation());

      return true;
    }

    bool VisitCXXRecordDecl(CXXRecordDecl* D) {
      if (!D->hasAttr<AnnotateAttr>())
        return true;

      if (ClassTemplateDecl* TmplD = D->getDescribedClassTemplate())
        return VisitTemplateDecl(TmplD);
      return true;
    }

    bool VisitTemplateTypeParmDecl(TemplateTypeParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool VisitTemplateDecl(TemplateDecl* D) {
      if (D->getTemplatedDecl() &&
          !D->getTemplatedDecl()->hasAttr<AnnotateAttr>())
        return true;

      // If we have a definition we might be about to re-#include the
      // same header containing definition that was #included previously,
      // i.e. we might have multiple fwd decls for the same template.
      // DO NOT remove the defaults here; the definition needs to keep it.
      // (ROOT-7037)
      if (ClassTemplateDecl* CTD = dyn_cast<ClassTemplateDecl>(D))
        if (CXXRecordDecl* TemplatedD = CTD->getTemplatedDecl())
          if (TemplatedD->getDefinition())
            return true;

      for(auto P: D->getTemplateParameters()->asArray())
        TraverseDecl(P);
      return true;
    }

    bool VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (D->hasDefaultArgument())
        D->removeDefaultArgument();
      return true;
    }

    bool VisitParmVarDecl(ParmVarDecl* D) {
      if (m_IsStoringState)
        return true;

      if (D->hasDefaultArg())
        D->setDefaultArg(nullptr);
      return true;
    }

    bool VisitEnumDecl(EnumDecl* D) {
      if (m_IsStoringState)
        return true;

      // Now that we will read the full enum, unload the forward decl.
      TransactionUnloader Unloader(m_Sema, 0);
      Unloader.UnloadDecl(D);
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
    // If File is 0 this means that the #included file doesn't exist.
    if (!File)
      return;

    auto found = m_Map.find(File);
    if (found == m_Map.end())
     return; // nothing to do, file not referred in any annotation

    AutoloadingVisitor defaultArgsCleaner;
    for (auto D : found->second) {
      defaultArgsCleaner.RemoveDefaultArgsOf(D, &getInterpreter()->getSema());
    }
    // Don't need to keep track of cleaned up decls from file.
    m_Map.erase(found);
  }

  AutoloadCallback::~AutoloadCallback() {
  }

  void AutoloadCallback::TransactionCommitted(const Transaction &T) {
    if (T.decls_begin() == T.decls_end())
      return;
    if (T.decls_begin()->m_DGR.isNull())
      return;

    // The first decl must be
    //   extern int __Cling_Autoloading_Map;
    bool HaveAutoloadingMapMarker = false;
    for (auto I = T.decls_begin(), E = T.decls_end();
         !HaveAutoloadingMapMarker && I != E; ++I) {
      if (I->m_Call != cling::Transaction::kCCIHandleTopLevelDecl)
        return;
      for (auto&& D: I->m_DGR) {
        if (isa<EmptyDecl>(D))
          continue;
        else if (auto VD = dyn_cast<VarDecl>(D)) {
          HaveAutoloadingMapMarker
            = VD->hasExternalStorage() && VD->getIdentifier()
              && VD->getName().equals("__Cling_Autoloading_Map");
          if (!HaveAutoloadingMapMarker)
            return;
          break;
        } else
          return;
      }
    }

    if (!HaveAutoloadingMapMarker)
      return;

    AutoloadingVisitor defaultArgsStateCollector;
    Preprocessor& PP = m_Interpreter->getCI()->getPreprocessor();
    for (auto I = T.decls_begin(), E = T.decls_end(); I != E; ++I)
      for (auto&& D: I->m_DGR)
        defaultArgsStateCollector.TrackDefaultArgStateOf(D, m_Map, PP);
  }

} //end namespace cling
