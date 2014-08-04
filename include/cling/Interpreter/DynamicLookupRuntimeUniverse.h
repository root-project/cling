//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_DYNAMIC_LOOKUP_RUNTIME_UNIVERSE_H
#define CLING_DYNAMIC_LOOKUP_RUNTIME_UNIVERSE_H

#ifndef __CLING__
#error "This file must not be included by compiled programs."
#endif

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/DynamicExprInfo.h"
#include "cling/Interpreter/DynamicLookupLifetimeHandler.h"
#include "cling/Interpreter/Value.h"

namespace cling {

/// \brief Contains declarations for cling's runtime.
namespace runtime {
  extern Interpreter* gCling;

  /// \brief Provides builtins, which are neccessary for the dynamic scopes
  /// and runtime bindings. These builtins should be used for other purposes.
  namespace internal {

    /// \brief EvaluateT is used to replace all invalid source code that
    /// occurs, when cling's dynamic extensions are enabled.
    ///
    /// When the interpreter "sees" invalid code it marks it and skip all the
    /// semantic checks (like with templates). Afterwords all these marked
    /// nodes are replaced with a call to EvaluateT, which makes valid
    /// C++ code. It is templated because it can be used in expressions and
    /// T is the type of the evaluated expression.
    ///
    /// @tparam T The type of the evaluated expression.
    /// @param[in] ExprInfo Helper structure that keeps information about the
    /// expression that is being replaced and the addresses of the variables
    /// that the replaced expression contains.
    /// @param[in] DC The declaration context, in which the expression will be
    /// evaluated at runtime.
    template<typename T>
    T EvaluateT(DynamicExprInfo* ExprInfo, clang::DeclContext* DC ) {
      Value result(
        cling::runtime::gCling->Evaluate(ExprInfo->getExpr(), DC,
                                         ExprInfo->isValuePrinterRequested())
                            );
      if (result.isValid())
        // Check whether the expected return type and the actual return type are
        // compatible with Sema::CheckAssingmentConstraints or
        // ASTContext::typesAreCompatible.
        return result.simplisticCastAs<T>();
      return T();
    }

    /// \brief EvaluateT specialization for the case where we instantiate with
    /// void.
    template<>
    void EvaluateT(DynamicExprInfo* ExprInfo, clang::DeclContext* DC ) {
      cling::runtime::gCling->Evaluate(ExprInfo->getExpr(), DC,
                       ExprInfo->isValuePrinterRequested());
    }
  } // end namespace internal
} // end namespace runtime
} // end namespace cling

#endif // CLING_DYNAMIC_LOOKUP_RUNTIME_UNIVERSE_H
