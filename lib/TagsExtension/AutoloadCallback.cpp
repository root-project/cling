#include "cling/TagsExtension/Callback.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Path.h"

namespace cling {

  bool AutoloadCallback::LookupObject (clang::LookupResult &R, clang::Scope *){
    std::string in=R.getLookupName().getAsString();
    clang::Sema& sema= m_Interpreter->getSema();

    unsigned id = sema.getDiagnostics().getCustomDiagID
            (clang::DiagnosticsEngine::Level::Warning,
                "Note: '%0' can be found in %1");
    unsigned idn = sema.getDiagnostics().getCustomDiagID
            (clang::DiagnosticsEngine::Level::Note,
                "Type : %0 , Full Path: %1");

    for (auto it = m_Tags->begin(in); it != m_Tags->end(in); ++it)
    {
      auto lookup = it->second;
      clang::SourceLocation loc = R.getNameLoc();

      if (loc.isInvalid())
          continue;

      sema.Diags.Report(R.getNameLoc(),id)
              << lookup.name
              << llvm::sys::path::filename(lookup.header);

      sema.Diags.Report(R.getNameLoc(),idn)
              << lookup.type
              << lookup.header;

    }
    return false;
  }


  AutoloadCallback::AutoloadCallback
  (cling::Interpreter* interp, TagManager *t) :
    InterpreterCallbacks(interp,true),
    m_Interpreter(interp),
    m_Tags(t) {
      //TODO : Invoke stdandard c++ tagging here
  }
  TagManager* AutoloadCallback::getTagManager() {
    return m_Tags;
  }

}//end namespace cling
