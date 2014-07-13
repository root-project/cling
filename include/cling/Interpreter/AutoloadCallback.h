#ifndef CLING_AUTOLOAD_CALLBACK_H
#define CLING_AUTOLOAD_CALLBACK_H

#include "cling/Interpreter/InterpreterCallbacks.h"
#include <map>

#if 0
This feature is disabled by default until stable.
To enable, execute the following code as runtime input.
Note that, for now, the T meta command will cause the interpreter to segfault,
unless these objects are loaded.

.rawInput 0
#include "cling/Interpreter/AutoloadCallback.h"
gCling->setCallbacks(new cling::AutoloadCallback(gCling));

#endif

namespace clang {
  class Decl;
  class ClassTemplateDecl;
  class NamespaceDecl;
}

namespace cling {
  class Interpreter;
  class Transaction;
}

namespace cling {
  class AutoloadCallback : public cling::InterpreterCallbacks {
  public:
      AutoloadCallback(cling::Interpreter* interp);
      ~AutoloadCallback();
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
    void TransactionCommitted(const Transaction& T);

  private:
    struct FileInfo {
      FileInfo():Included(false){}
      bool Included;
      std::vector<clang::Decl*> Decls;
    };

    // The key is the Unique File ID obtained from the source manager.
    std::map<unsigned,FileInfo> m_Map;

    Interpreter* m_Interpreter;
//    AutoloadingStateInfo m_State;

    void report(clang::SourceLocation l, std::string name,std::string header);
    void InsertIntoAutoloadingState(clang::Decl* decl,std::string annotation);
    void HandleDeclVector(std::vector<clang::Decl*> Decls);
    void HandleNamespace(clang::NamespaceDecl* NS);
    void HandleClassTemplate(clang::ClassTemplateDecl* CT);
  };
} // end namespace cling

#endif // CLING_AUTOLOAD_CALLBACK_H

