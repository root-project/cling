//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_H
#define CLING_VALUE_H

#include <stddef.h>
#include <assert.h>

namespace llvm {
  class Type;
  struct GenericValue;
}
namespace clang {
  class ASTContext;
  class QualType;
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
  protected:
    /// \brief value
    char /*llvm::GenericValue*/ m_GV[48]; // 48 bytes on 64bit

    /// \brief the value's type according to clang, stored as void* to reduce
    /// dependencies.
    void* /*clang::QualType*/ m_ClangType;

    /// \brief the value's type according to clang
    const llvm::Type* m_LLVMType;

    enum EStorageType {
      kSignedIntegerOrEnumerationType,
      kUnsignedIntegerOrEnumerationType,
      kDoubleType,
      kFloatType,
      kLongDoubleType,
      kPointerType,
      kUnsupportedType
    };

    /// \brief Retrieve the underlying, canonical, desugared, unqualified type.
    EStorageType getStorageType() const;

  public:

    /// \brief Default constructor, creates a value that IsInvalid().
    Value();
    Value(const Value& other);
    /// \brief Construct a valid Value.
    Value(const llvm::GenericValue& v, clang::QualType t);

    Value(const llvm::GenericValue& v, clang::QualType clangTy, 
          const llvm::Type* llvmTy);

    Value& operator =(const Value& other);

    llvm::GenericValue getGV() const;
    void setGV(llvm::GenericValue GV);
    clang::QualType getClangType() const;
    const llvm::Type* getLLVMType() const { return m_LLVMType; }
    void setLLVMType(const llvm::Type* Ty) { m_LLVMType = Ty; }

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const;

    /// \brief Determine whether the Value is set but void.
    bool isVoid(const clang::ASTContext& ASTContext) const;

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    bool hasValue(const clang::ASTContext& ASTContext) const {
      return isValid() && !isVoid(ASTContext); }

    /// \brief Get the value without type checking.
    template <typename T>
    T getAs() const;

    template <typename T>
    T* getAs(T**) const { return (T*)getAs((void**)0); }
    void* getAs(void**) const;
    double getAs(double*) const;
    long double getAs(long double*) const;
    float getAs(float*) const;
    bool getAs(bool*) const;
    signed char getAs(signed char*) const;
    unsigned char getAs(unsigned char*) const;
    signed short getAs(signed short*) const;
    unsigned short getAs(unsigned short*) const;
    signed int getAs(signed int*) const;
    unsigned int getAs(unsigned int*) const;
    signed long getAs(signed long*) const;
    unsigned long getAs(unsigned long*) const;
    signed long long getAs(signed long long*) const;
    unsigned long long getAs(unsigned long long*) const;

    /// \brief Get the value.
    //
    /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
    /// casting the value of builtins (except void), enums and pointers.
    /// Values referencing an object are treated as pointers to the object.
    template <typename T>
    T simplisticCastAs() const;
  };


  template <typename T>
  T Value::getAs() const {
    // T *must* correspond to type. Else use simplisticCastAs()!
    return getAs((T*)0);
  }
  template <typename T>
  T Value::simplisticCastAs() const {
    EStorageType storageType = getStorageType();
    switch (storageType) {
    case kSignedIntegerOrEnumerationType:
      return (T) getAs<signed long long>();
    case kUnsignedIntegerOrEnumerationType:
      return (T) getAs<unsigned long long>();
    case kDoubleType:
      return (T) getAs<double>();
    case kFloatType:
      return (T) getAs<float>();
    case kLongDoubleType:
      return (T) getAs<long double>();
    case kPointerType:
      return (T) (size_t) getAs<void*>();
    case kUnsupportedType:
      assert("unsupported type in Value, cannot cast simplistically!" && 0);
    }
    return T();
  }
} // end namespace cling

#endif // CLING_VALUE_H
