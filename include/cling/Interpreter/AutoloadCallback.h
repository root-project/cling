#ifndef CLING_AUTOLOAD_CALLBACK_H
#define CLING_AUTOLOAD_CALLBACK_H

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

#if 0
This feature is disabled by default until stable.
To enable, execute the following code as runtime input.
Note that, for now, the T meta command will cause the interpreter to segfault,
unless these objects are loaded.

.rawInput 0
#include "cling/Interpreter/AutoloadCallback.h"
gCling->setCallbacks(new cling::AutoloadCallback(gCling));

#endif

namespace cling {
  class AutoloadCallback : public cling::InterpreterCallbacks {
  public:
      AutoloadCallback(cling::Interpreter* interp);
    
//    using cling::InterpreterCallbacks::LookupObject;
      //^to get rid of bogus warning : "-Woverloaded-virtual"
      //virtual functions ARE meant to be overriden!

//    bool LookupObject (clang::LookupResult &R, clang::Scope *);
//    bool LookupObject (clang::TagDecl* t);

    void InclusionDirective(clang::SourceLocation HashLoc,
                            const clang::Token &IncludeTok,
                            llvm::StringRef FileName,
                            bool IsAngled,
                            clang::CharSourceRange FilenameRange,
                            const clang::FileEntry *File,
                            llvm::StringRef SearchPath,
                            llvm::StringRef RelativePath,
                            const clang::Module *Imported);

  private:
    Interpreter* m_Interpreter;


    void report(clang::SourceLocation l, std::string name,std::string header);
  };
} // end namespace cling

#endif // CLING_AUTOLOAD_CALLBACK_H

