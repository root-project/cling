//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_DYNAMIC_LOOKUP_LIFETIME_HANDLER_H
#define CLING_DYNAMIC_LOOKUP_LIFETIME_HANDLER_H

#include <string>

namespace clang {
  class DeclContext;
}

namespace cling {
  class Interpreter;

/// \brief Contains declarations for cling's runtime.
namespace runtime {

  /// \brief Provides private definitions for the dynamic scopes and runtime
  /// bindings. These builtins should not be used for other purposes.
  namespace internal {
    class DynamicExprInfo;

    /// \brief LifetimeHandler is used in case of initialization using address
    /// on the automatic store (stack) instead of EvaluateT.
    ///
    /// The reason is to avoid the copy constructors that might be private.
    /// This is part of complex transformation, which aims to preserve the
    /// code behavior. For example:
    /// @code
    /// int i = 5;
    /// MyClass my(dep->Symbol(i))
    /// @endcode
    /// where dep->Symbol() is a symbol not known at compile-time
    /// transformed into:
    /// @code
    /// cling::runtime::internal::LifetimeHandler
    /// __unique("dep->Sybmol(*(int*)@)",(void*[]){&i}, DC, "MyClass");
    /// MyClass &my(*(MyClass*)__unique.getMemory());
    /// @endcode
    class LifetimeHandler {
    private:
      /// \brief The Interpreter handling this storage element.
      Interpreter* m_Interpreter;

      /// \brief The memory on the free store, where the object will be
      /// created.
      void* m_Memory;

      /// \brief The type of the object that will be created.
      std::string m_Type;

    public:
      /// \brief Constructs an expression, which creates the object on the
      /// free store and tells the interpreter to evaluate it.
      ///
      /// @param[in] ExprInfo Helper structure that keeps information about
      /// the expression that is being replaced and the addresses of the
      /// variables that the replaced expression contains.
      /// @param[in] DC The declaration context, in which the expression will
      /// be evaluated at runtime
      /// @param[in] type The type of the object, which will help to delete
      /// it, when the LifetimeHandler goes out of scope.
      /// @param[in] Interp The current interpreter object, which evaluate will
      /// be called upon.
      ///
      LifetimeHandler(DynamicExprInfo* ExprInfo,
                      clang::DeclContext* DC,
                      const char* type,
                      Interpreter* Interp);

      ///\brief Returns the created object.
      inline void* getMemory() const { return m_Memory; }

      /// \brief Clears up the free store, when LifetimeHandler goes out of
      /// scope.
      ///
      ~LifetimeHandler();
    };
  }
} // end namespace runtime
} // end namespace cling

#endif // CLING_DYNAMIC_LOOKUP_LIFETIME_HANDLER_H
