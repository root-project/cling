//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Manasij Mukherjee  <manasij7479@gmail.com>
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AUTOLOAD_CALLBACK_H
#define CLING_AUTOLOAD_CALLBACK_H

#include "cling/Interpreter/InterpreterCallbacks.h"

#include "llvm/ADT/DenseMap.h"

namespace clang {
  class Decl;
  class ClassTemplateDecl;
  class NamespaceDecl;
  class FunctionDecl;
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
    using cling::InterpreterCallbacks::LookupObject;
      //^to get rid of bogus warning : "-Woverloaded-virtual"
      //virtual functions ARE meant to be overriden!

//    bool LookupObject (clang::LookupResult &R, clang::Scope *);
    bool LookupObject (clang::TagDecl* t);

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

    typedef llvm::DenseMap<const clang::FileEntry*, std::vector<clang::Decl*> > FwdDeclsMap;
  private:
    // The key is the Unique File ID obtained from the source manager.
    FwdDeclsMap m_Map;

    Interpreter* m_Interpreter;
//    AutoloadingStateInfo m_State;

    void report(clang::SourceLocation l, std::string name,std::string header);
  };
} // end namespace cling

#endif // CLING_AUTOLOAD_CALLBACK_H

