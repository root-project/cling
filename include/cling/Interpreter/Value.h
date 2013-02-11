//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_H
#define CLING_VALUE_H

#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"

#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Type.h"

namespace clang {
  class ASTContext;
}

namespace cling {
  ///\brief A type, value pair.
  //
  /// Type-safe value access and setting. Simple (built-in) casting is
  /// available, but better extract the value using the template
  /// parameter that matches the Value's type.
  ///
  /// The class represents a llvm::GenericValue with its corresponding
  /// clang::QualType. Use-cases:
  /// 1. Expression evaluation: we need to know the type of the GenericValue
  /// that we have gotten from the JIT
  /// 2. Value printer: needs to know the type in order to skip the printing of
  /// void types
  /// 3. Parameters for calls given an llvm::Function and a clang::FunctionDecl.
  class Value {
  private:
    /// \brief Forward decl for typed access specializations
    template <typename T> struct TypedAccess;

  protected:
    /// \brief value
    llvm::GenericValue m_GV;

    /// \brief the value's type according to clang
    clang::QualType m_ClangType;

    /// \brief the value's type according to clang
    const llvm::Type* m_LLVMType;

  public:

    /// \brief Default constructor, creates a value that IsInvalid().
    Value() {}
    /// \brief Construct a valid Value.
    Value(const llvm::GenericValue& v, clang::QualType t) 
      : m_GV(v), m_ClangType(t), m_LLVMType(0) { }

    Value(const llvm::GenericValue& v, clang::QualType clangTy, 
          const llvm::Type* llvmTy) 
      : m_GV(v), m_ClangType(clangTy), m_LLVMType(llvmTy) { }

    llvm::GenericValue getGV() const { return m_GV; }
    void setGV(llvm::GenericValue GV) { m_GV = GV; }
    clang::QualType getClangType() const { return m_ClangType; }
    const llvm::Type* getLLVMType() const { return m_LLVMType; }
    void setLLVMType(const llvm::Type* Ty) { m_LLVMType = Ty; }

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const { return !m_ClangType.isNull(); }

    /// \brief Determine whether the Value is set but void.
    bool isVoid(const clang::ASTContext& ASTContext) const {
      return isValid() 
        && m_ClangType.getDesugaredType(ASTContext)->isVoidType();
    }

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    bool hasValue(const clang::ASTContext& ASTContext) const {
      return isValid() && !isVoid(ASTContext); }

    /// \brief Get the value without type checking.
    template <typename T>
    T getAs() const;

    /// \brief Get the value.
    //
    /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
    /// casting the value of builtins (except void), enums and pointers.
    /// Values referencing an object are treated as pointers to the object.
    template <typename T>
    T simplisticCastAs() const;
  };

  template<typename T>
  struct Value::TypedAccess{
    T extract(const llvm::GenericValue& value) {
      return *reinterpret_cast<T*>(value.PointerVal);
    }
  };
  template<typename T>
  struct Value::TypedAccess<T*>{
    T* extract(const llvm::GenericValue& value) {
      return reinterpret_cast<T*>(value.PointerVal);
    }
  };

#define CLING_VALUE_TYPEDACCESS(TYPE, GETTER)       \
  template<>                                        \
  struct Value::TypedAccess<TYPE> {                 \
    TYPE extract(const llvm::GenericValue& value) { \
      return value.GETTER;                          \
    }                                               \
  }

#define CLING_VALUE_TYPEDACCESS_SIGNED(TYPE)       \
  CLING_VALUE_TYPEDACCESS(signed TYPE, IntVal.getSExtValue())

#define CLING_VALUE_TYPEDACCESS_UNSIGNED(TYPE)     \
  CLING_VALUE_TYPEDACCESS(unsigned TYPE, IntVal.getZExtValue())

#define CLING_VALUE_TYPEDACCESS_BOTHSIGNS(TYPE)     \
  CLING_VALUE_TYPEDACCESS_SIGNED(TYPE);             \
  CLING_VALUE_TYPEDACCESS_UNSIGNED(TYPE);

  CLING_VALUE_TYPEDACCESS(double, DoubleVal);
  //CLING_VALUE_TYPEDACCESS(long double, ???);
  CLING_VALUE_TYPEDACCESS(float, FloatVal);

  CLING_VALUE_TYPEDACCESS(bool, IntVal.getBoolValue());

  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(char)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(short)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(int)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(long)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(long long)

#undef CLING_VALUE_TYPEDACCESS_BOTHSIGNS
#undef CLING_VALUE_TYPEDACCESS_UNSIGNED
#undef CLING_VALUE_TYPEDACCESS_SIGNED
#undef CLING_VALUE_TYPEDACCESS

  template <typename T>
  T Value::getAs() const {
    // T *must* correspond to type. Else use simplisticCastAs()!
    TypedAccess<T> VI;
    return VI.extract(m_GV);
  }
  template <typename T>
  T Value::simplisticCastAs() const {
    const clang::Type* desugCanon = m_ClangType->getUnqualifiedDesugaredType();
    desugCanon = desugCanon->getCanonicalTypeUnqualified()->getTypePtr()
       ->getUnqualifiedDesugaredType();
    if (desugCanon->isSignedIntegerOrEnumerationType()) {
      return (T) getAs<signed long long>();
    } else if (desugCanon->isUnsignedIntegerOrEnumerationType()) {
      return (T) getAs<unsigned long long>();
    } else if (desugCanon->isRealFloatingType()) {
      const clang::BuiltinType* BT = desugCanon->getAs<clang::BuiltinType>();
      if (BT->getKind() == clang::BuiltinType::Double)
        return (T) getAs<double>();
      else if (BT->getKind() == clang::BuiltinType::Float)
        return (T) getAs<float>();
      /* not yet supported in JIT:
      else if (BT->getKind() == clang::BuiltinType::LongDouble)
        return (T) getAs<long double>();
      */
    } else if (desugCanon->isPointerType() || desugCanon->isObjectType()) {
      return (T) (size_t) getAs<void*>();
    }
    assert("unsupported type in Value, cannot cast simplistically!" && 0);
    return T();
  }
} // end namespace cling

#endif // CLING_VALUE_H
