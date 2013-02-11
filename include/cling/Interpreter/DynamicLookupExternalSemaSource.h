//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/InterpreterCallbacks.h"

namespace cling {

  class Interpreter;

  /// \brief Provides last chance of recovery for clang semantic analysis.
  /// When the compiler doesn't find the symbol in its symbol table it asks
  /// its ExternalSemaSource to look for the symbol.
  ///
  /// In contrast to the compiler point of view, where these symbols must be
  /// errors, the interpreter's point of view these symbols are to be
  /// evaluated at runtime. For that reason the interpreter marks all unknown
  /// by the compiler symbols to be with delayed lookup (evaluation).
  /// One have to be carefull in the cases, in which the compiler expects that
  /// the lookup will fail!
  class DynamicIDHandler : public InterpreterExternalSemaSource {
  public:
    DynamicIDHandler(InterpreterCallbacks* C) 
      : InterpreterExternalSemaSource(C) { }
    ~DynamicIDHandler();

    /// \brief Provides last resort lookup for failed unqualified lookups
    ///
    /// If there is failed lookup, tell sema to create an artificial declaration
    /// which is of dependent type. So the lookup result is marked as dependent
    /// and the diagnostics are suppressed. After that is's an interpreter's
    /// responsibility to fix all these fake declarations and lookups.
    /// It is done by the DynamicExprTransformer.
    ///
    /// \param[out] R The recovered symbol.
    /// \param[in] S The scope in which the lookup failed.
    ///
    /// \returns true if the name was found and the compilation should continue.
    ///
    virtual bool LookupUnqualified(clang::LookupResult& R, clang::Scope* S);

    /// \brief Checks whether this name should be marked as dynamic, i.e. for
    /// runtime resolution.
    ///
    /// \param[out] R The recovered symbol.
    /// \param[in] S The scope in which the lookup failed.
    ///
    /// \returns true if the name should be marked as dynamic.
    ///
    static bool IsDynamicLookup(clang::LookupResult& R, clang::Scope* S);
  };
} // end namespace cling
