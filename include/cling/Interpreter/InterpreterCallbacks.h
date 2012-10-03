//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INTERPRETER_CALLBACKS_H
#define CLING_INTERPRETER_CALLBACKS_H

namespace clang {
  class LookupResult;
  class Scope;
}

namespace cling {
  class Interpreter;
  class Transaction;

  /// \brief  This interface provides a way to observe the actions of the
  /// interpreter as it does its thing.  Clients can define their hooks here to
  /// implement interpreter level tools.
  class InterpreterCallbacks {
  private:
    // The callbacks should contain the interpreter in case of more than one
    InterpreterCallbacks(){}

  protected:
    Interpreter* m_Interpreter;
    bool m_Enabled;

  public:
    InterpreterCallbacks(Interpreter* interp, bool enabled = false)
      : m_Interpreter(interp) {
      setEnabled(enabled);
    }

    virtual ~InterpreterCallbacks() {}

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
    virtual bool LookupObject(clang::LookupResult&, clang::Scope*) {
      return false;
    }

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
    virtual void TransactionUnloaded(const Transaction&) {};
  };
} // end namespace cling

#endif // CLING_INTERPRETER_CALLBACKS_H
