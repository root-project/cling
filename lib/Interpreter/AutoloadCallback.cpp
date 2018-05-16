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
#include "cling/Utils/Output.h"
#include "DeclUnloader.h"



#include <clang/Lex/HeaderSearch.h>

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
    /// removing the default template arguments (that will now be provided by
    /// the definition read from the include).
    bool m_IsStoringState;
    bool m_IsAutloadEntry;  // True during the traversal of an explicitly annotated decl.
    AutoloadCallback::FwdDeclsMap* m_Map;
    clang::Preprocessor* m_PP;
    clang::Sema* m_Sema;

    std::pair<const clang::FileEntry*,const clang::FileEntry*> m_PrevFE;
    std::pair<std::string,std::string> m_PrevFileName;
  private:
    bool IsAutoloadEntry(Decl *D) {
      for(auto attr = D->specific_attr_begin<AnnotateAttr>(),
               end = D->specific_attr_end<AnnotateAttr> ();
          attr != end;
          ++attr)
      {
        //        cling::errs() << "Annotation: " << c->getAnnotation() << "\n";
        if (!attr->isInherited()) {
          llvm::StringRef annotation = attr->getAnnotation();
          assert(!annotation.empty() && "Empty annotation!");
          if (annotation.startswith(llvm::StringRef(annoTag, lenAnnoTag))) {
            // autoload annotation.
            return true;
          }
        }
      }
      return false;
    }

    using Annotations_t = std::pair<llvm::StringRef,llvm::StringRef>;

    void InsertIntoAutoloadingState(Decl* decl, Annotations_t FileNames) {

      assert(m_PP);

      auto addFile = [this,decl](llvm::StringRef FileName, bool warn) {
        if (FileName.empty()) return (const FileEntry*)nullptr;

        const FileEntry* FE = 0;
        SourceLocation fileNameLoc;
        // Remember this file wth full path, not "./File.h" (ROOT-8863).
        bool isAngled = true;
        const DirectoryLookup* FromDir = 0;
        const FileEntry* FromFile = 0;
        const DirectoryLookup* CurDir = 0;
        bool needCacheUpdate = false;

        if (FileName.equals(m_PrevFileName.first))
          FE = m_PrevFE.first;
        else if (FileName.equals(m_PrevFileName.second))
          FE = m_PrevFE.second;
        else {
          FE = m_PP->LookupFile(fileNameLoc, FileName, isAngled,
                                FromDir, FromFile, CurDir, /*SearchPath*/0,
                                /*RelativePath*/ 0, /*suggestedModule*/0,
                                /*IsMapped*/0, /*SkipCache*/ false,
                                /*OpenFile*/ false, /*CacheFail*/ true);
          needCacheUpdate = true;
        }

        if (FE) {
          auto& Vec = (*m_Map)[FE];
          Vec.push_back(decl);
          if (needCacheUpdate) return FE;
          else return (const FileEntry*)nullptr;
        } else if (warn) {
          // If the top level header is expected to be findable at run-time,
          // the direct header might not because the include path might be
          // different enough and only the top-header is guaranteed to be seen
          // by the user as an interface header to be available on the
          // run-time include path.
          cling::errs()
          << "Error in cling::AutoloadingVisitor::InsertIntoAutoloadingState:\n"
          "   Missing FileEntry for " << FileName << "\n";
          if (NamedDecl* ND = dyn_cast<NamedDecl>(decl)) {
            cling::errs() << "   requested to autoload type ";
            ND->getNameForDiagnostic(cling::errs(),
                                     ND->getASTContext().getPrintingPolicy(),
                                     true /*qualified*/);
            cling::errs() << "\n";
          }
          return (const FileEntry*)nullptr;
        } else {
          // Case of the direct header that is not a top level header, no
          // warning in this case (to likely to be a false positive).
          return (const FileEntry*)nullptr;
        }
      };

      const FileEntry* cacheUpdate;

      if ( (cacheUpdate = addFile(FileNames.first,true)) ) {
        m_PrevFE.first = cacheUpdate;
        m_PrevFileName.first = FileNames.first;
      }
      if ( (cacheUpdate = addFile(FileNames.second,false)) ) {
        m_PrevFE.second = cacheUpdate;
        m_PrevFileName.second = FileNames.second;
      }


    }

  public:
    AutoloadingVisitor():
      m_IsStoringState(false), m_IsAutloadEntry(false), m_Map(0), m_PP(0),
    m_Sema(0), m_PrevFE({nullptr,nullptr})
    {}

    void RemoveDefaultArgsOf(Decl* D, Sema* S) {
      m_Sema = S;

      auto cursor = D->getMostRecentDecl();
      m_IsAutloadEntry = IsAutoloadEntry(cursor);
      TraverseDecl(cursor);
      while (cursor != D && (cursor = cursor->getPreviousDecl())) {
        m_IsAutloadEntry = IsAutoloadEntry(cursor);
        TraverseDecl(cursor);
      }
      m_IsAutloadEntry = false;
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

      Annotations_t annotations;
      for(auto attr = D->specific_attr_begin<AnnotateAttr> (),
          end = D->specific_attr_end<AnnotateAttr> ();
          attr != end;
          ++attr)
      {
        if (!attr->isInherited()) {
          auto annot = attr->getAnnotation();
          if (annot.startswith(llvm::StringRef(annoTag, lenAnnoTag))) {
            if (annotations.first.empty()) {
              annotations.first = annot.drop_front(lenAnnoTag);
            } else {
              annotations.second = annot.drop_front(lenAnnoTag);
            }
          }
        }
      }
      InsertIntoAutoloadingState(D, annotations);

      return true;
    }

    bool VisitCXXRecordDecl(CXXRecordDecl* D) {
      // Since we are only interested in fixing forward declaration
      // there is no need to continue on when we see a complete definition.
      if (D->isCompleteDefinition())
        return false;

      if (!D->hasAttr<AnnotateAttr>())
        return true;

      if (ClassTemplateDecl* TmplD = D->getDescribedClassTemplate())
        return VisitTemplateDecl(TmplD);
      return true;
    }

    bool VisitTemplateTypeParmDecl(TemplateTypeParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (m_IsAutloadEntry) {
        if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
          D->removeDefaultArgument();
      } else {
        if (D->hasDefaultArgument() && D->defaultArgumentWasInherited())
          D->removeDefaultArgument();
      }
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

      if (m_IsAutloadEntry) {
        if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
          D->removeDefaultArgument();
      } else {
        if (D->hasDefaultArgument() && D->defaultArgumentWasInherited())
          D->removeDefaultArgument();
      }
      return true;
    }

    bool VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl* D) {
      if (m_IsStoringState)
        return true;

      if (m_IsAutloadEntry) {
        if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
          D->removeDefaultArgument();
      } else {
        if (D->hasDefaultArgument() && D->defaultArgumentWasInherited())
          D->removeDefaultArgument();
      }
      return true;
    }

    bool VisitParmVarDecl(ParmVarDecl* D) {
      if (m_IsStoringState)
        return true;

      if (m_IsAutloadEntry) {
        if (D->hasDefaultArg() && !D->hasInheritedDefaultArg())
          D->setDefaultArg(nullptr);
      } else {
        if (D->hasDefaultArg() && D->hasInheritedDefaultArg())
          D->setDefaultArg(nullptr);
      }
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
