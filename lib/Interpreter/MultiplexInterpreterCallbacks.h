//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Manasij Mukherjee  <manasij7479@gmail.com>
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_MULTIPLEX_INTERPRETER_CALLBACKS_H
#define CLING_MULTIPLEX_INTERPRETER_CALLBACKS_H

#include "cling/Interpreter/InterpreterCallbacks.h"

namespace cling {

  class MultiplexInterpreterCallbacks : public InterpreterCallbacks {
  private:
    std::vector<InterpreterCallbacks*> m_Callbacks;

  public:
    MultiplexInterpreterCallbacks(Interpreter* interp)
      : InterpreterCallbacks(interp, true, true, true) {}

    void addCallback(InterpreterCallbacks* newCb) {
      m_Callbacks.push_back(newCb);
    }

    void InclusionDirective(clang::SourceLocation HashLoc,
                            const clang::Token& IncludeTok,
                            llvm::StringRef FileName, bool IsAngled,
                            clang::CharSourceRange FilenameRange,
                            const clang::FileEntry* File,
                            llvm::StringRef SearchPath,
                            llvm::StringRef RelativePath,
                            const clang::Module* Imported) {
      for (InterpreterCallbacks* cb : m_Callbacks)
        cb->InclusionDirective(HashLoc, IncludeTok, FileName, IsAngled,
                               FilenameRange, File, SearchPath, RelativePath,
                               Imported);
    }

    bool FileNotFound(llvm::StringRef FileName,
                               llvm::SmallVectorImpl<char>& RecoveryPath) {
      bool result = false;
      for (InterpreterCallbacks* cb : m_Callbacks)
        result = cb->FileNotFound(FileName, RecoveryPath) || result;
      return result;
    }

     bool LookupObject(clang::LookupResult& LR, clang::Scope* S) {
       bool result = false;
       for (InterpreterCallbacks* cb : m_Callbacks)
         result = cb->LookupObject(LR, S) || result;
       return result;
     }

     bool LookupObject(const clang::DeclContext* DC,  clang::DeclarationName DN) {
       bool result = false;
       for (InterpreterCallbacks* cb : m_Callbacks)
         result = cb->LookupObject(DC, DN) || result;
       return result;
     }

     bool LookupObject(clang::TagDecl* T) {
       bool result = false;
       for (InterpreterCallbacks* cb : m_Callbacks)
         result = cb->LookupObject(T) || result;
       return result;
     }

     void TransactionCommitted(const Transaction& T) {
       for (InterpreterCallbacks* cb : m_Callbacks) {
         cb->TransactionCommitted(T);
       }
     }

     void TransactionUnloaded(const Transaction& T) {
       for (InterpreterCallbacks* cb : m_Callbacks) {
         cb->TransactionUnloaded(T);
       }
     }

     void DeclDeserialized(const clang::Decl* D) {
       for (InterpreterCallbacks* cb : m_Callbacks) {
         cb->DeclDeserialized(D);
       }
     }

     void TypeDeserialized(const clang::Type* Ty) {
       for (InterpreterCallbacks* cb : m_Callbacks) {
         cb->TypeDeserialized(Ty);
       }
     }

     void LibraryLoaded(const void* Lib, llvm::StringRef Name) {
       for (InterpreterCallbacks* cb : m_Callbacks) {
         cb->LibraryLoaded(Lib,Name);
       }
     }

     void LibraryUnloaded(const void* Lib, llvm::StringRef Name) {
       for (InterpreterCallbacks* cb : m_Callbacks) {
         cb->LibraryUnloaded(Lib,Name);
       }
     }
  };
} // end namespace cling

#endif // CLING_MULTIPLEX_INTERPRETER_CALLBACKS_H
