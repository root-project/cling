//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INTERPRETER_CALLBACKS_H
#define CLING_INTERPRETER_CALLBACKS_H

#include "clang/AST/DeclarationName.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>

namespace clang {
  class ASTDeserializationListener;
  class Decl;
  class DeclContext;
  class DeclarationName;
  class ExternalSemaSource;
  class LookupResult;
  class NamedDecl;
  class Scope;
  class TagDecl;
  class Type;
  class Token;
  class FileEntry;
  class Module;
}

namespace cling {
  class Interpreter;
  class InterpreterCallbacks;
  class InterpreterDeserializationListener;
  class InterpreterExternalSemaSource;
  class InterpreterPPCallbacks;
  class Transaction;

  /// \brief  This interface provides a way to observe the actions of the
  /// interpreter as it does its thing.  Clients can define their hooks here to
  /// implement interpreter level tools.
  class InterpreterCallbacks {
  protected:

    ///\brief Our interpreter instance.
    ///
    Interpreter* m_Interpreter; // we don't own

    ///\brief Our custom SemaExternalSource, translating interesting events into
    /// callbacks. RefOwned by Sema & ASTContext.
    ///
    InterpreterExternalSemaSource* m_ExternalSemaSource;

    ///\brief Our custom ASTDeserializationListener, translating interesting
    /// events into callbacks.
    ///
    std::unique_ptr
    <InterpreterDeserializationListener> m_DeserializationListener;

    ///\brief Our custom PPCallbacks, translating interesting
    /// events into interpreter callbacks.
    ///
    InterpreterPPCallbacks* m_PPCallbacks;

    ///\brief DynamicScopes only! Set to true only when evaluating dynamic expr.
    ///
    bool m_IsRuntime;

  protected:
    void UpdateWithNewDecls(const clang::DeclContext *DC,
                            clang::DeclarationName Name,
                            llvm::ArrayRef<clang::NamedDecl*> Decls);
  public:

    ///\brief Constructs the callbacks with default callback adaptors.
    ///
    ///\param[in] interp - an interpreter.
    ///\param[in] enableExternalSemaSourceCallbacks  - creates a default
    ///           InterpreterExternalSemaSource and attaches it to Sema.
    ///\param[in] enableDeserializationListenerCallbacks - creates a default
    ///           InterpreterDeserializationListener and attaches it to the
    ///           ModuleManager if it is set.
    ///\param[in] enablePPCallbacks  - creates a default InterpreterPPCallbacks
    ///           and attaches it to the Preprocessor.
    ///
    InterpreterCallbacks(Interpreter* interp,
                         bool enableExternalSemaSourceCallbacks = false,
                         bool enableDeserializationListenerCallbacks = false,
                         bool enablePPCallbacks = false);

    virtual ~InterpreterCallbacks();

    cling::Interpreter* getInterpreter() const { return m_Interpreter; }
    clang::ExternalSemaSource* getInterpreterExternalSemaSource() const;

    clang::ASTDeserializationListener*
    getInterpreterDeserializationListener() const;

   virtual void InclusionDirective(clang::SourceLocation /*HashLoc*/,
                                   const clang::Token& /*IncludeTok*/,
                                   llvm::StringRef FileName,
                                   bool /*IsAngled*/,
                                   clang::CharSourceRange /*FilenameRange*/,
                                   const clang::FileEntry* /*File*/,
                                   llvm::StringRef /*SearchPath*/,
                                   llvm::StringRef /*RelativePath*/,
                                   const clang::Module* /*Imported*/) {}
    virtual void EnteredSubmodule(clang::Module* M,
                                  clang::SourceLocation ImportLoc,
                                  bool ForPragma) {}

    virtual bool FileNotFound(llvm::StringRef FileName,
                              llvm::SmallVectorImpl<char>& RecoveryPath);

    /// \brief This callback is invoked whenever the interpreter needs to
    /// resolve the type and the adress of an object, which has been marked for
    /// delayed evaluation from the interpreter's dynamic lookup extension.
    ///
    /// \returns true if lookup result is found and should be used.
    ///
    // FIXME: Find a way to merge the three of them.
    virtual bool LookupObject(clang::LookupResult&, clang::Scope*);
    virtual bool LookupObject(const clang::DeclContext*, clang::DeclarationName);
    virtual bool LookupObject(clang::TagDecl*);

    /// \brief This callback is invoked whenever the interpreter failed to load a library.
    ///
    /// \param[in] - Error message and parameters passed to loadLibrary
    /// \returns true if the error was handled.
    virtual bool LibraryLoadingFailed(const std::string&, const std::string&, bool, bool) { return 0; }

    ///\brief This callback is invoked whenever interpreter has committed new
    /// portion of declarations.
    ///
    ///\param[in] - The transaction that was committed.
    ///
    virtual void TransactionCommitted(const Transaction&) {}

    ///\brief This callback is invoked whenever interpreter has reverted a
    /// transaction that has been fully committed.
    ///
    ///\param[in] - The transaction that was reverted.
    ///
    virtual void TransactionUnloaded(const Transaction&) {}

    ///\brief This callback is invoked whenever a transaction is rolled back.
    ///
    ///\param[in] - The transaction that was reverted.
    ///
    virtual void TransactionRollback(const Transaction&) {}

    /// \brief Used to inform client about a new decl read by the ASTReader.
    ///
    ///\param[in] - The Decl read by the ASTReader.
    virtual void DeclDeserialized(const clang::Decl*) {}

    /// \brief Used to inform client about a new type read by the ASTReader.
    ///
    ///\param[in] - The Type read by the ASTReader.
    virtual void TypeDeserialized(const clang::Type*) {}

    virtual void LibraryLoaded(const void*, llvm::StringRef) {}
    virtual void LibraryUnloaded(const void*, llvm::StringRef) {}

    ///\brief Cling calls this is printing a stack trace can be beneficial,
    /// for instance when throwing interpreter exceptions.
    virtual void PrintStackTrace() {}

    ///\brief Interface to support locking the interpreter state in case of
    /// concurrent usage.
    ///
    /// Cling assumes that any of its function is invoked in a locked context,
    /// but before invoking user code (e.g. static initialization or value
    /// printing) cling will calling `EnteringUserCode()`, and once
    /// done call `ReturnedFromUserCode()`. Typically the user provided locks
    /// would be unlock by `EnteringUserCode()` and lock back in
    /// `ReturnedFromUserCode()`. State can be returned from EnteringUserCode
    /// and made use of in ReturnedFromUserCode(), to identify pairs of these
    /// calls.
    virtual void* EnteringUserCode() { return nullptr; }

    ///\brief See `EnteringFromUserCode()`!
    virtual void ReturnedFromUserCode(void*) {}

    ///\brief Lock a region of compilation that is executed by the interpreter
    /// during user code execution.
    ///
    /// When cling is used in multi-threaded environments, all calls to cling
    /// are expected to be locked by the caller. Cling will release that lock
    /// using `EnteringUserCode()` and re-instate that lock using
    /// `ReturnedFromUserCode()` for the duration of the execution of the user
    /// code. But that user code can trigger calls to the interpreter itself.
    /// These calls are due to instrumented parts of the user code, e.g.
    /// `printValue()` calls and `cling::runtime::internal::LifetimeHandler`
    /// calls. For those, cling needs to be locked with a mechanism compatible
    /// with the mechanism used for `EnteringUserCode()` /
    /// `ReturnedFromUserCode()` to avoid deadlocks. Before entering compilation
    /// triggered by user code, cling will call
    /// `LockCompilationDuringUserCodeExecution()`; after the execution of that
    /// code has finished it will call
    /// `UnlockCompilationDuringUserCodeExecution()`.
    /// Note that after the compilation of that code cling will call
    /// `EnteringUserCode()` (before executing) and `ReturnedFromUserCode()`
    /// (after execution that code).
    ///
    /// \returns An optional state object needed for the call to
    /// `UnlockCompilationDuringUserCodeExecution(state)`.
    virtual void* LockCompilationDuringUserCodeExecution() { return nullptr; }

    /// \brief Unlocks recursive compilation; see the documentation of
    /// `LockCompilationDuringUserCodeExecution()`.
    virtual void UnlockCompilationDuringUserCodeExecution(void* /*StateInfo*/) {}

    ///\brief DynamicScopes only! Set to true if it is currently evaluating a
    /// dynamic expr.
    ///
    virtual void SetIsRuntime(bool val);
  };
} // end namespace cling

// TODO: Make the build system in the testsuite aware how to build that class
// and extract it out there again.
namespace cling {
  namespace test {
    class TestProxy {
    public:
      TestProxy();
      int Draw();
      const char* getVersion();

      int Add(int a, int b);
      int Add10(int num);
      void PrintString(std::string s);
      bool PrintArray(int a[], size_t size);

      void PrintArray(float a[][5], size_t size);

      void PrintArray(int a[][4][5], size_t size);
    };

    extern TestProxy* Tester;

    class SymbolResolverCallback: public cling::InterpreterCallbacks {
    private:
      bool m_Resolve;
      clang::NamedDecl* m_TesterDecl;
    public:
      SymbolResolverCallback(Interpreter* interp, bool resolve = true);
      ~SymbolResolverCallback();

      bool LookupObject(clang::LookupResult& R, clang::Scope* S);
      bool LookupObject(const clang::DeclContext*, clang::DeclarationName) {
        return false;
      }
      bool LookupObject(clang::TagDecl* Tag) {
        return false;
      }
      bool ShouldResolveAtRuntime(clang::LookupResult& R, clang::Scope* S);
    };
  } // end test
} // end cling

#endif // CLING_INTERPRETER_CALLBACKS_H
