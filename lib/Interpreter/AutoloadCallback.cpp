#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/AST.h"
#include "clang/Lex/Preprocessor.h"

#include "AutoloadingStateInfo.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/AutoloadCallback.h"


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

  void AutoloadCallback::InclusionDirective(clang::SourceLocation HashLoc,
                          const clang::Token &IncludeTok,
                          llvm::StringRef FileName,
                          bool IsAngled,
                          clang::CharSourceRange FilenameRange,
                          const clang::FileEntry *File,
                          llvm::StringRef SearchPath,
                          llvm::StringRef RelativePath,
                          const clang::Module *Imported) {

    auto iterator = m_Interpreter->getAutoloadingState()->m_Map.find(File->getUID());
    if (iterator == m_Interpreter->getAutoloadingState()->m_Map.end())
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

    }

  }

  AutoloadCallback::AutoloadCallback(Interpreter* interp) :
    InterpreterCallbacks(interp,true,false,true), m_Interpreter(interp){
  }

}//end namespace cling
