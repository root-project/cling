#include "cling/TagsExtension/AutoloadCallback.h"
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

  bool AutoloadCallback::LookupObject (LookupResult &R, Scope *) {
    std::string in=R.getLookupName().getAsString();
    Sema& sema= m_Interpreter->getSema();

    unsigned id
      = sema.getDiagnostics().getCustomDiagID (DiagnosticsEngine::Level::Warning,
                                               "Note: '%0' can be found in %1");
    unsigned idn
      = sema.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Level::Note,
                                              "Type : %0 , Full Path: %1");

    for (auto it = m_Tags->begin(in); it != m_Tags->end(in); ++it) {
      auto lookup = it->second;
      SourceLocation loc = R.getNameLoc();

      if (loc.isInvalid())
          continue;

      sema.Diags.Report(R.getNameLoc(), id)
        << lookup.name
        << llvm::sys::path::filename(lookup.header);

      sema.Diags.Report(R.getNameLoc(),idn)
        << lookup.type
        << lookup.header;

    }
    return false;
  }

  AutoloadCallback::AutoloadCallback(Interpreter* interp, TagManager *t) :
    InterpreterCallbacks(interp,true), m_Interpreter(interp), m_Tags(t) {
    //TODO : Invoke stdandard c++ tagging here
    // FIXME: There is an m_Interpreter in the base class InterpreterCallbacks.
  }

  TagManager* AutoloadCallback::getTagManager() {
    return m_Tags;
  }

}//end namespace cling
