#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/AST.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/AutoloadCallback.h"
#include "cling/Interpreter/DynamicLibraryManager.h"
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

//  bool AutoloadCallback::LookupObject (TagDecl *t) {
//    if (t->hasAttr<AnnotateAttr>())
//      report(t->getLocation(),t->getNameAsString(),t->getAttr<AnnotateAttr>()->getAnnotation());
//    return false;
//  }

  void removeDefaultArg(TemplateParameterList *Params) {
    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      Decl *Param = Params->getParam(i);
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
              if(TTP->hasDefaultArgument())
                TTP->setDefaultArgument(nullptr,false);
      }
      else if(NonTypeTemplateParmDecl *NTTP =
                dyn_cast<NonTypeTemplateParmDecl>(Param)) {
                  if(NTTP->hasDefaultArgument())
                    NTTP->setDefaultArgument(nullptr,false);
      }
    }
  }
  void removeDefaultArg(FunctionDecl* fd) {
    for (unsigned i = 0, e = fd->getNumParams(); i != e; ++i) {
      if(fd->getParamDecl(i)->hasDefaultArg()) {
        fd->getParamDecl(i)->setDefaultArg(nullptr);
      }
    }
  }

  void AutoloadCallback::InclusionDirective(clang::SourceLocation HashLoc,
                          const clang::Token &IncludeTok,
                          llvm::StringRef FileName,
                          bool IsAngled,
                          clang::CharSourceRange FilenameRange,
                          const clang::FileEntry *File,
                          llvm::StringRef SearchPath,
                          llvm::StringRef RelativePath,
                          const clang::Module *Imported) {
    if (!File)
      return;
    auto iterator = m_Map.find(File->getUID());
    if (iterator == m_Map.end())
      return; // nothing to do, file not referred in any annotation
    if(iterator->second.Included)
      return; // nothing to do, file already included once

    iterator->second.Included = true;

    for(clang::Decl* decl : iterator->second.Decls) {
//      llvm::outs() <<"CB:"<<llvm::cast<NamedDecl>(decl)->getName() <<"\n";
      decl->dropAttrs();
      if(llvm::isa<clang::EnumDecl>(decl)) {
        //do something (remove fixed type..how?)
      }
      if(llvm::isa<clang::ClassTemplateDecl>(decl)) {
        clang::ClassTemplateDecl* ct = llvm::cast<clang::ClassTemplateDecl>(decl);
//        llvm::outs() << ct->getName() <<"\n";
        removeDefaultArg(ct->getTemplateParameters());
      }
      if(llvm::isa<clang::FunctionDecl>(decl)) {
        clang::FunctionDecl* fd = llvm::cast<clang::FunctionDecl>(decl);
        removeDefaultArg(fd);
      }

    }

  }

  AutoloadCallback::AutoloadCallback(Interpreter* interp) :
    InterpreterCallbacks(interp,true,false,true), m_Interpreter(interp){

  }
  AutoloadCallback::~AutoloadCallback() {
  }

  void AutoloadCallback::InsertIntoAutoloadingState
    (clang::Decl* decl,std::string annotation) {
      std::string canonicalFile = DynamicLibraryManager::normalizePath(annotation);

      if (canonicalFile.empty())
        canonicalFile = annotation;

      clang::Preprocessor& PP = m_Interpreter->getCI()->getPreprocessor();
      const FileEntry* FE = 0;
      SourceLocation fileNameLoc;
      bool isAngled = false;
      const DirectoryLookup* LookupFrom = 0;
      const DirectoryLookup* CurDir = 0;

      FE = PP.LookupFile(fileNameLoc, canonicalFile, isAngled, LookupFrom, CurDir,
                        /*SearchPath*/0, /*RelativePath*/ 0,
                        /*suggestedModule*/0, /*SkipCache*/false,
                        /*OpenFile*/ false, /*CacheFail*/ false);

      if(!FE)
        return;

      auto& stateMap = m_Map;
      auto iterator = stateMap.find(FE->getUID());

      if(iterator == stateMap.end())
        stateMap[FE->getUID()] = FileInfo();

      stateMap[FE->getUID()].Decls.push_back(decl);

  }

  void AutoloadCallback::HandleNamespace(NamespaceDecl* NS) {
    std::vector<clang::Decl*> decls;
    for(auto dit = NS->decls_begin(); dit != NS->decls_end(); ++dit)
      decls.push_back(*dit);
    HandleDeclVector(decls);
  }

  void AutoloadCallback::HandleClassTemplate(ClassTemplateDecl* CT) {
    CXXRecordDecl* cxxr = CT->getTemplatedDecl();
    if(cxxr->hasAttr<AnnotateAttr>())
      InsertIntoAutoloadingState(CT,cxxr->getAttr<AnnotateAttr>()->getAnnotation());
  }
  void AutoloadCallback::HandleFunction(FunctionDecl *F) {
    if(F->hasAttr<AnnotateAttr>())
      InsertIntoAutoloadingState(F,F->getAttr<AnnotateAttr>()->getAnnotation());
  }

  void AutoloadCallback::HandleDeclVector(std::vector<clang::Decl*> Decls) {
    for(auto decl : Decls ) {
      if(auto ct = llvm::dyn_cast<ClassTemplateDecl>(decl))
        HandleClassTemplate(ct);
      if(auto ns = llvm::dyn_cast<NamespaceDecl>(decl))
        HandleNamespace(ns);
      if(auto f = llvm::dyn_cast<FunctionDecl>(decl))
        HandleFunction(f);
    }
  }
  void AutoloadCallback::TransactionCommitted(const Transaction &T) {
      for (Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
           I != E; ++I) {
        Transaction::DelayCallInfo DCI = *I;
        std::vector<clang::Decl*> decls;
        for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
               JE = DCI.m_DGR.end(); J != JE; ++J) {

          decls.push_back(*J);
        }
        HandleDeclVector(decls);
      }
  }


}//end namespace cling
