#ifndef CLING_CTAGS_AUTOLOAD_CALLBACK_H
#define CLING_CTAGS_AUTOLOAD_CALLBACK_H

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/TagsExtension/TagManager.h"
#include "clang/Sema/Lookup.h"

#if 0
This feature is disabled by default until stable.
To enable, execute the following code as runtime input.
Note that, for now, the T meta command will cause the interpreter to segfault,
unless these objects are loaded.

.rawInput 0
#include "cling/TagsExtension/TagManager.h"
#include "cling/TagsExtension/AutoloadCallback.h"
cling::TagManager t;
gCling->setCallbacks(new cling::AutoloadCallback(gCling,&t));

#endif

namespace cling {
  class LookupInfo{};
  //TODO: Would contain info regarding previous lookups;
  //TODO: get rid of the map in LookupObject
  
  class AutoloadCallback : public cling::InterpreterCallbacks {
  public:
      AutoloadCallback(cling::Interpreter* interp, cling::TagManager* t);
    
    using cling::InterpreterCallbacks::LookupObject;
      //^to get rid of bogus warning : "-Woverloaded-virtual"
      //virtual functions ARE meant to be overriden!

    bool LookupObject (clang::LookupResult &R, clang::Scope *);
    bool LookupObject (clang::TagDecl* t);
    
    TagManager* getTagManager();
  private:
    Interpreter* m_Interpreter;
    TagManager* m_Tags;

    void report(clang::SourceLocation l, std::string name,std::string header);
  };
} // end namespace cling

#endif // CLING_CTAGS_AUTOLOAD_CALLBACK_H

