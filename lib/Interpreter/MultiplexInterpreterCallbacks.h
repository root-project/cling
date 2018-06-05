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
    std::vector<std::unique_ptr<InterpreterCallbacks>> m_Callbacks;

  public:
    MultiplexInterpreterCallbacks(Interpreter* interp)
      : InterpreterCallbacks(interp, true, true, true) {}

    void addCallback(std::unique_ptr<InterpreterCallbacks> newCb) {
      m_Callbacks.push_back(std::move(newCb));
    }

    void InclusionDirective(clang::SourceLocation HashLoc,
                            const clang::Token& IncludeTok,
                            llvm::StringRef FileName, bool IsAngled,
                            clang::CharSourceRange FilenameRange,
                            const clang::FileEntry* File,
                            llvm::StringRef SearchPath,
                            llvm::StringRef RelativePath,
                            const clang::Module* Imported) override {
      for (auto&& cb : m_Callbacks)
        cb->InclusionDirective(HashLoc, IncludeTok, FileName, IsAngled,
                               FilenameRange, File, SearchPath, RelativePath,
                               Imported);
    }

    bool FileNotFound(llvm::StringRef FileName,
                      llvm::SmallVectorImpl<char>& RecoveryPath) override {
      bool result = false;
      for (auto&& cb : m_Callbacks)
        result = cb->FileNotFound(FileName, RecoveryPath) || result;
      return result;
    }

     bool LookupObject(clang::LookupResult& LR, clang::Scope* S) override {
       bool result = false;
       for (auto&& cb : m_Callbacks)
         result = cb->LookupObject(LR, S) || result;
       return result;
     }

     bool LookupObject(const clang::DeclContext* DC,
                       clang::DeclarationName DN) override {
       bool result = false;
       for (auto&& cb : m_Callbacks)
         result = cb->LookupObject(DC, DN) || result;
       return result;
     }

     bool LookupObject(clang::TagDecl* T) override {
       bool result = false;
       for (auto&& cb : m_Callbacks)
         result = cb->LookupObject(T) || result;
       return result;
     }

     void TransactionCommitted(const Transaction& T) override {
       for (auto&& cb : m_Callbacks) {
         cb->TransactionCommitted(T);
       }
     }

     bool LibraryLoadingFailed(const std::string& errmessage, const std::string& libStem, bool permanent,
         bool resolved) override {
       for (auto&& cb : m_Callbacks) {
         if (bool res = cb->LibraryLoadingFailed(errmessage, libStem, permanent, resolved))
           return res;
       }
       return 0;
     }

     void TransactionUnloaded(const Transaction& T) override {
       for (auto&& cb : m_Callbacks) {
         cb->TransactionUnloaded(T);
       }
     }

     void TransactionRollback(const Transaction& T) override {
        for (auto&& cb : m_Callbacks) {
           cb->TransactionRollback(T);
        }
     }

     void DeclDeserialized(const clang::Decl* D) override {
       for (auto&& cb : m_Callbacks) {
         cb->DeclDeserialized(D);
       }
     }

     void TypeDeserialized(const clang::Type* Ty) override {
       for (auto&& cb : m_Callbacks) {
         cb->TypeDeserialized(Ty);
       }
     }

     void LibraryLoaded(const void* Lib, llvm::StringRef Name) override {
       for (auto&& cb : m_Callbacks) {
         cb->LibraryLoaded(Lib,Name);
       }
     }

     void LibraryUnloaded(const void* Lib, llvm::StringRef Name) override {
       for (auto&& cb : m_Callbacks) {
         cb->LibraryUnloaded(Lib,Name);
       }
     }

    void SetIsRuntime(bool val) override {
      InterpreterCallbacks::SetIsRuntime(val);
      for (auto&& cb : m_Callbacks)
        cb->SetIsRuntime(val);
    }

    void PrintStackTrace() override {
      for (auto&& cb : m_Callbacks)
        cb->PrintStackTrace();
    }

    void* EnteringUserCode() override {
      void* ret = nullptr;
      for (auto&& cb : m_Callbacks) {
        if (void* myret = cb->EnteringUserCode()) {
          assert(!ret && "Multiple state infos are not supported!");
          ret = myret;
        }
      }
      return ret;
    }

    void ReturnedFromUserCode(void* StateInfo) override {
      for (auto&& cb : m_Callbacks)
        cb->ReturnedFromUserCode(StateInfo);
    }

    void* LockCompilationDuringUserCodeExecution() override {
      void* ret = nullptr;
      for (auto&& cb : m_Callbacks) {
        if (void* myret = cb->LockCompilationDuringUserCodeExecution()) {
          // FIXME: it'd be better to introduce a new Callback interface type
          // that does not allow multiplexing, and thus enforces that there
          // is only one single listener.
          assert(!ret && "Multiple state infos are not supported!");
          ret = myret;
        }
      }
      return ret;
    }

    void UnlockCompilationDuringUserCodeExecution(void* StateInfo) override {
      for (auto&& cb : m_Callbacks)
        cb->UnlockCompilationDuringUserCodeExecution(StateInfo);
    }
  };
} // end namespace cling

#endif // CLING_MULTIPLEX_INTERPRETER_CALLBACKS_H
