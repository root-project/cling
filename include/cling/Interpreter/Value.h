//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_H
#define CLING_VALUE_H

namespace clang {
  class ASTContext;
  class QualType;
  class RecordDecl;
}

namespace cling {
  class Interpreter;

  ///\brief A type, value pair.
  //
  /// Type-safe value access and setting. Simple (built-in) casting is
  /// available, but better extract the value using the template
  /// parameter that matches the Value's type.
  ///
  /// The class represents a llvm::GenericValue with its corresponding
  /// clang::QualType. Use-cases are expression evaluation, value printing
  /// and parameters for function calls.
  class Value {
  protected:
    ///\brief Multi-purpose storage.
    ///
    union Storage {
      long long m_LL;
      unsigned long long m_ULL;
      void* m_Ptr; /// Can point to allocation, see needsManagedAllocation().
      float m_Float;
      double m_Double;
      long double m_LongDouble;
    };

    /// \brief The actual value.
    Storage m_Storage;

    /// \brief The value's type, stored as opaque void* to reduce
    /// dependencies.
    void* m_Type;

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

    /// \brief Allocate storage as needed by the type.
    void ManagedAllocate(Interpreter* interp);

    /// \brief Assert in case of an unsupported type. Outlined to reduce include
    ///   dependencies.
    void AssertOnUnsupportedTypeCast() const;

    /// \brief Get the function address of the wrapper of the destructor.
    void* GetDtorWrapperPtr(const clang::RecordDecl* RD,
                            Interpreter& interp) const;

    unsigned long GetNumberOfElements() const;

  public:
    /// \brief Default constructor, creates a value that IsInvalid().
    Value(): m_Type(0) {}
    /// \brief Copy a value.
    Value(const Value& other);
    /// \brief Move a value.
    Value(Value&& other);
    /// \brief Construct a valid but ininitialized Value. After this call the
    ///   value's storage can be accessed; i.e. calls ManagedAllocate() if
    ///   needed.
    Value(clang::QualType Ty, Interpreter* Interp);
    /// \brief Destruct the value; calls ManagedFree() if needed.
    ~Value();

    Value& operator =(const Value& other);
    Value& operator =(Value&& other);

    clang::QualType getType() const;

    /// \brief Whether this type needs managed heap, i.e. the storage provided
    /// by the m_Storage member is insufficient, or a non-trivial destructor
    /// must be called.
    bool needsManagedAllocation() const;

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const;

    /// \brief Determine whether the Value is set but void.
    bool isVoid(const clang::ASTContext& Ctx) const;

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    bool hasValue(const clang::ASTContext& Ctx) const {
      return isValid() && !isVoid(Ctx); }

    /// \brief Get a reference to the value without type checking.
    /// T *must* correspond to type. Else use simplisticCastAs()!
    template <typename T>
    T& getAs() { return getAs((T*)0); }

    /// \brief Get the value without type checking.
    /// T *must* correspond to type. Else use simplisticCastAs()!
    template <typename T>
    T getAs() const { return const_cast<Value*>(this)->getAs<T>(); }

    template <typename T>
    T*& getAs(T**) const { return (T*&)getAs((void**)0); }
    void*& getAs(void**) { return m_Storage.m_Ptr; }
    double& getAs(double*) { return m_Storage.m_Double; }
    long double& getAs(long double*) { return m_Storage.m_LongDouble; }
    float& getAs(float*) { return m_Storage.m_Float; }
    long long& getAs(long long*) { return m_Storage.m_LL; }
    unsigned long long& getAs(unsigned long long*) { return m_Storage.m_ULL; }

    void*& getPtr() { return m_Storage.m_Ptr; }
    double& getDouble() { return m_Storage.m_Double; }
    long double& getLongDouble() { return m_Storage.m_LongDouble; }
    float& getFloat() { return m_Storage.m_Float; }
    long long& getLL() { return m_Storage.m_LL; }
    unsigned long long& getULL() { return m_Storage.m_ULL; }

    void* getPtr() const { return m_Storage.m_Ptr; }
    double getDouble() const { return m_Storage.m_Double; }
    long double getLongDouble() const { return m_Storage.m_LongDouble; }
    float getFloat() const { return m_Storage.m_Float; }
    long long getLL() const { return m_Storage.m_LL; }
    unsigned long long getULL() const { return m_Storage.m_ULL; }

    /// \brief Get the value with cast.
    //
    /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
    /// casting the value of builtins (except void), enums and pointers.
    /// Values referencing an object are treated as pointers to the object.
    template <typename T>
    T simplisticCastAs() const;
  };

  template <typename T>
  T Value::simplisticCastAs() const {
    EStorageType storageType = getStorageType();
    switch (storageType) {
    case kSignedIntegerOrEnumerationType:
      return (T) getAs<long long>();
    case kUnsignedIntegerOrEnumerationType:
      return (T) getAs<unsigned long long>();
    case kDoubleType:
      return (T) getAs<double>();
    case kFloatType:
      return (T) getAs<float>();
    case kLongDoubleType:
      return (T) getAs<long double>();
    case kPointerType:
      return (T) (unsigned long) getAs<void*>();
    case kUnsupportedType:
      AssertOnUnsupportedTypeCast();
    }
    return T();
  }
} // end namespace cling

#endif // CLING_VALUE_H
