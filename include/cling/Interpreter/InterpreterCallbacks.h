//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INTERPRETER_CALLBACKS_H
#define CLING_INTERPRETER_CALLBACKS_H

#include "clang/Sema/ExternalSemaSource.h"

#include "llvm/ADT/OwningPtr.h"

namespace clang {
  class LookupResult;
  class Scope;
}

namespace cling {
  class Interpreter;
  class InterpreterCallbacks;
  class InterpreterExternalSemaSource;
  class Transaction;

  ///\brief Translates 'interesting' for the interpreter ExternalSemaSource 
  /// events into interpreter callbacks.
  ///
  class InterpreterExternalSemaSource : public clang::ExternalSemaSource {
  private:

    ///\brief The interpreter callback which are subscribed for the events.
    ///
    /// Usually the callbacks is the owner of the class and the interpreter owns
    /// the callbacks so they can't be out of sync. Eg we notifying the wrong
    /// callback class.
    ///
    InterpreterCallbacks* m_Callbacks; // we don't own it.

  public:
    InterpreterExternalSemaSource(InterpreterCallbacks* cb) : m_Callbacks(cb){}

    ~InterpreterExternalSemaSource();

    /// \brief Provides last resort lookup for failed unqualified lookups.
    ///
    /// This gets translated into InterpreterCallback's call.
    ///
    ///\param[out] R The recovered symbol.
    ///\param[in] S The scope in which the lookup failed.
    ///
    ///\returns true if a suitable declaration is found.
    ///
    virtual bool LookupUnqualified(clang::LookupResult& R, clang::Scope* S);
  };

  /// \brief  This interface provides a way to observe the actions of the
  /// interpreter as it does its thing.  Clients can define their hooks here to
  /// implement interpreter level tools.
  class InterpreterCallbacks {
  private:
    // The callbacks should contain the interpreter in case of more than one
    InterpreterCallbacks(){}

  protected:

    ///\brief Our interpreter instance.
    ///
    Interpreter* m_Interpreter; // we don't own

    ///\brief Whether or not the callbacks are enabled.
    ///
    bool m_Enabled;
    
    ///\brief Our custom SemaExternalSource, translating interesting events into
    /// callbacks.
    ///
    llvm::OwningPtr<InterpreterExternalSemaSource> m_SemaExternalSource;
  public:
    InterpreterCallbacks(Interpreter* interp, bool enabled = false);

    virtual ~InterpreterCallbacks();

    void setEnabled(bool e = true) {
      m_Enabled = e;
    }

    bool isEnabled() { return m_Enabled; }

    /// \brief This callback is invoked whenever the interpreter needs to
    /// resolve the type and the adress of an object, which has been marked for
    /// delayed evaluation from the interpreter's dynamic lookup extension.
    ///
    /// \returns true if lookup result is found and should be used.
    ///
    virtual bool LookupObject(clang::LookupResult&, clang::Scope*);

    ///\brief This callback is invoked whenever interpreter has committed new
    /// portion of declarations.
    ///
    ///\param[out] - The transaction that was committed.
    ///
    virtual void TransactionCommitted(const Transaction&) {}

    ///\brief This callback is invoked whenever interpreter has reverted a
    /// portion of declarations.
    ///
    ///\param[out] - The transaction that was reverted.
    ///
    virtual void TransactionUnloaded(const Transaction&) {}
  };
} // end namespace cling

#endif // CLING_INTERPRETER_CALLBACKS_H
