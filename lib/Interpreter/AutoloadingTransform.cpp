#include "clang/Sema/Sema.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/AST.h"

#include "AutoloadingTransform.h"
#include "AutoloadingStateInfo.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/DynamicLibraryManager.h"

using namespace clang;

namespace cling {
  AutoloadingTransform::AutoloadingTransform(clang::Sema* S, Interpreter *ip)
    : TransactionTransformer(S),m_Interpreter(ip) {
  }

  AutoloadingTransform::~AutoloadingTransform()
  {}

  void InsertIntoAutoloadingState(Interpreter* interp,clang::Decl* decl,std::string annotation) {
      std::string canonicalFile = DynamicLibraryManager::normalizePath(annotation);

      if (canonicalFile.empty())
        canonicalFile = annotation;

      clang::Preprocessor& PP = interp->getCI()->getPreprocessor();
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
      auto& stateMap = interp->getAutoloadingState()->m_Map;
      auto iterator = stateMap.find(FE->getUID());

      if(iterator == stateMap.end())
        stateMap[FE->getUID()] = AutoloadingStateInfo::FileInfo();

      stateMap[FE->getUID()].Decls.push_back(decl);

  }


  void HandleDeclVector(std::vector<clang::Decl*> Decls, Interpreter* interp);
  void HandleNamespace(NamespaceDecl* NS,Interpreter* interp) {
    std::vector<clang::Decl*> decls;
    for(auto dit = NS->decls_begin(); dit != NS->decls_end(); ++dit)
      decls.push_back(*dit);
    HandleDeclVector(decls,interp);
  }

  void HandleClassTemplate(ClassTemplateDecl* CT,Interpreter* interp) {
    CXXRecordDecl* cxxr = CT->getTemplatedDecl();
    if(cxxr->hasAttr<AnnotateAttr>())
      InsertIntoAutoloadingState(interp,CT,cxxr->getAttr<AnnotateAttr>()->getAnnotation());
  }

  void HandleDeclVector(std::vector<clang::Decl*> Decls, Interpreter* interp) {
    for(auto decl : Decls ) {
      if(auto ct = llvm::dyn_cast<ClassTemplateDecl>(decl))
        HandleClassTemplate(ct,interp);
      if(auto ns = llvm::dyn_cast<NamespaceDecl>(decl))
        HandleNamespace(ns,interp);
    }
  }


  void AutoloadingTransform::Transform() {
    const Transaction* T = getTransaction();
    for (Transaction::const_iterator I = T->decls_begin(), E = T->decls_end();
         I != E; ++I) {
      Transaction::DelayCallInfo DCI = *I;
      std::vector<clang::Decl*> decls;
      for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
             JE = DCI.m_DGR.end(); J != JE; ++J) {

//FIXME: Enable when safe !
//        if ( (*J)->hasAttr<AnnotateAttr>() /*FIXME: && CorrectCallbackLoaded() how ? */  )
//          clang::Decl::castToDeclContext(*J)->setHasExternalLexicalStorage();

        decls.push_back(*J);
      }
      HandleDeclVector(decls,m_Interpreter);
    }
  }
} // end namespace cling
