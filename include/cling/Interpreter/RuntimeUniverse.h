//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_RUNTIME_UNIVERSE_H
#define CLING_RUNTIME_UNIVERSE_H

#if !defined(__CLING__)
#error "This file must not be included by compiled programs."
#endif

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS // needed by System/DataTypes.h
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS // needed by System/DataTypes.h
#endif

#ifdef __cplusplus

#include "cling/Interpreter/RuntimeException.h"

#include <new>

namespace cling {

  class Interpreter;

  /// \brief Used to stores the declarations, which are going to be
  /// available only at runtime. These are cling runtime builtins
  namespace runtime {

    /// \brief The interpreter provides itself as a builtin, i.e. it
    /// interprets itself. This is particularly important for implementing
    /// the dynamic scopes and the runtime bindings
    extern Interpreter* gCling;

    namespace internal {
      /// \brief Some of clang's routines rely on valid source locations and
      /// source ranges. This member can be looked up and source locations and
      /// ranges can be passed in as parameters to these routines.
      ///
      /// Use instead of SourceLocation() and SourceRange(). This might help,
      /// when clang emits diagnostics on artificially inserted AST node.
      int InterpreterGeneratedCodeDiagnosticsMaybeIncorrect;


      ///\brief Set the type of a void expression evaluated at the prompt.
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpSVR - The Value that is created.
      ///
      void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn);

      ///\brief Set the value of the GenericValue for the expression
      /// evaluated at the prompt.
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] value - The float value of the assignment to be stored
      ///                    in GenericValue.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpSVR - The Value that is created.
      ///
      void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char vpOn,
                           float value);

      ///\brief Set the value of the GenericValue for the expression
      /// evaluated at the prompt.
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] value - The double value of the assignment to be stored
      ///                    in GenericValue.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpSVR - The Value that is created.
      ///
      void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char vpOn,
                           double value);

      ///\brief Set the value of the GenericValue for the expression
      ///   evaluated at the prompt. Extract through
      ///   APFloat(ASTContext::getFloatTypeSemantics(QT), const APInt &)
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] value - The value of the assignment to be stored
      ///                    in GenericValue.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpSVR - The Value that is created.
      ///
      void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char vpOn,
                           long double value);

      ///\brief Set the value of the GenericValue for the expression
      /// evaluated at the prompt.
      /// We are using unsigned long long instead of uint64, because we don't
      /// want to #include the header.
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] value - The uint64_t value of the assignment to be stored
      ///                    in GenericValue.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpSVR - The Value that is created.
      ///
      void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char vpOn,
                           unsigned long long value);

      ///\brief Set the value of the GenericValue for the expression
      /// evaluated at the prompt.
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] value - The void* value of the assignment to be stored
      ///                    in GenericValue.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpV - The Value that is created.
      ///
      void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char vpOn,
                           const void* value);

      ///\brief Set the value of the Generic value and return the address
      /// for the allocated storage space.
      ///\param [in] vpI - The cling::Interpreter for Value.
      ///\param [in] vpQT - The opaque ptr for the clang::QualType of value.
      ///\param [in] vpT - The opaque ptr for the cling::Transaction.
      ///\param [out] vpV - The Value that is created.
      ///
      ///\returns the address where the value should be put.
      ///
      void* setValueWithAlloc(void* vpI, void* vpV, void* vpQT, char vpOn);

      ///\brief Placement new doesn't work for arrays. It needs to be called on
      /// each element. For non-PODs we also need to call the *structors. This
      /// handles also multi dimension arrays since the init order is
      /// independent on the dimensions.
      ///
      /// We must be consistent with clang. Eg:
      ///\code
      ///extern "C" int printf(const char*,...);
      /// struct S {
      ///    static int sI;
      ///    int I;
      ///    S(): I(sI++) {}
      /// };
      /// int S::sI = 0;
      /// S arr[5][3];
      /// int main() {
      ///    for (int i = 0; i < 5; ++i)
      ///    for (int j = 0; j < 3; ++j)
      ///       printf("[%d][%d]%d\n", i, j, arr[i][j].I);
      ///    return 0;
      /// }
      ///\endcode
      /// must be consistent with what clang does, since it is not well defined
      /// in the C++ standard.
      ///
      ///\param[in] src - array to copy
      ///\param[in] placement - where to copy
      ///\param[in] size - size of the array.
      ///
      template <typename T>
      void copyArray(T* src, void* placement, int size) {
        for (int i = 0; i < size; ++i)
          new ((void*)(((T*)placement) + i)) T(src[i]);
      }
    } // end namespace internal
  } // end namespace runtime
} // end namespace cling

using namespace cling::runtime;

extern "C" {
  ///\brief a function that throws NullDerefException. This allows to 'hide' the
  /// definition of the exceptions from the RuntimeUniverse and allows us to
  /// run cling in -no-rtti mode.
  ///
  void cling__runtime__internal__throwNullDerefException(void* Sema,
                                                         void* Expr);

}
#endif // __cplusplus

#endif // CLING_RUNTIME_UNIVERSE_H
