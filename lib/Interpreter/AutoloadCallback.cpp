#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/AutoloadCallback.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Path.h"

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


  AutoloadCallback::AutoloadCallback(Interpreter* interp) :
    InterpreterCallbacks(interp,true), m_Interpreter(interp){
  }

}//end namespace cling
