#ifndef CLING_MULTIPLEX_INTERPRETER_CALLBACKS_H
#define CLING_MULTIPLEX_INTERPRETER_CALLBACKS_H
#include "InterpreterCallbacks.h"
namespace cling {
  class MultiplexInterpreterCallbacks : public InterpreterCallbacks {
  public:
    MultiplexInterpreterCallbacks(cling::Interpreter* interp)
        :InterpreterCallbacks(interp){}
    void addCallback(InterpreterCallbacks* newCb) {
      m_Callbacks.push_back(newCb);
    }
    void InclusionDirective(clang::SourceLocation HashLoc,
                                    const clang::Token& IncludeTok,
                                    llvm::StringRef FileName,
                                    bool IsAngled,
                                    clang::CharSourceRange FilenameRange,
                                    const clang::FileEntry* File,
                                    llvm::StringRef SearchPath,
                                    llvm::StringRef RelativePath,
                                    const clang::Module* Imported) {
      for(InterpreterCallbacks* cb : m_Callbacks)
        cb->InclusionDirective(HashLoc, IncludeTok,
                               FileName, IsAngled,
                               FilenameRange, File,
                               SearchPath, RelativePath, Imported);
    }

    bool FileNotFound(llvm::StringRef FileName,
                               llvm::SmallVectorImpl<char>& RecoveryPath) {
      bool result = false;
      for(InterpreterCallbacks* cb : m_Callbacks)
        result = result || cb->FileNotFound(FileName, RecoveryPath);
      return result;
    }

     bool LookupObject(clang::LookupResult& LR, clang::Scope* S) {
       bool result = false;
       for(InterpreterCallbacks* cb : m_Callbacks)
         result = result || cb->LookupObject(LR, S);
       return result;
     }

     bool LookupObject(const clang::DeclContext* DC,  clang::DeclarationName DN) {
       bool result = false;
       for(InterpreterCallbacks* cb : m_Callbacks)
         result = result || cb->LookupObject(DC, DN);
       return result;
     }

     bool LookupObject(clang::TagDecl* T) {
       bool result = false;
       for(InterpreterCallbacks* cb : m_Callbacks)
         result = result || cb->LookupObject(T);
       return result;
     }

     void TransactionCommitted(const Transaction& T) {
       for(InterpreterCallbacks* cb : m_Callbacks) {
         cb->TransactionCommitted(T);
       }
     }

     void TransactionUnloaded(const Transaction& T) {
       for(InterpreterCallbacks* cb : m_Callbacks) {
         cb->TransactionUnloaded(T);
       }
     }

     void DeclDeserialized(const clang::Decl* D) {
       for(InterpreterCallbacks* cb : m_Callbacks) {
         cb->DeclDeserialized(D);
       }
     }

     void TypeDeserialized(const clang::Type* Ty) {
       for(InterpreterCallbacks* cb : m_Callbacks) {
         cb->TypeDeserialized(Ty);
       }
     }

     void LibraryLoaded(const void* Lib, llvm::StringRef Name) {
       for(InterpreterCallbacks* cb : m_Callbacks) {
         cb->LibraryLoaded(Lib,Name);
       }
     }
     void LibraryUnloaded(const void* Lib, llvm::StringRef Name) {
       for(InterpreterCallbacks* cb : m_Callbacks) {
         cb->LibraryUnloaded(Lib,Name);
       }
     }


  private:
    std::vector<InterpreterCallbacks*> m_Callbacks;
  };
} // end namespace cling
#endif
