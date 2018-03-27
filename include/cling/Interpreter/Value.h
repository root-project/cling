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

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace llvm {
  class raw_ostream;
}

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
  ///
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

    enum EStorageType {
      kSignedIntegerOrEnumerationType,
      kUnsignedIntegerOrEnumerationType,
      kDoubleType,
      kFloatType,
      kLongDoubleType,
      kPointerType,
      kManagedAllocation,
      kUnsupportedType
    };

    /// \brief Which part in m_Storage is active.
    EStorageType m_StorageType;

    /// \brief The value's type, stored as opaque void* to reduce
    /// dependencies.
    void* m_Type;

    ///\brief Interpreter that produced the value.
    ///
    Interpreter* m_Interpreter;

    /// \brief Retrieve the underlying, canonical, desugared, unqualified type.
    EStorageType getStorageType() const { return m_StorageType; }

    /// \brief Determine the underlying, canonical, desugared, unqualified type:
    /// the element of Storage to be used.
    static EStorageType determineStorageType(clang::QualType QT);

    /// \brief Determine the underlying, canonical, desugared, unqualified type:
    /// the element of Storage to be used.
    static constexpr EStorageType determineStorageTypeT(...) {
      return kManagedAllocation;
    }

    template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    static constexpr EStorageType determineStorageTypeT(T*) {
      return std::is_signed<T>::value
        ? kSignedIntegerOrEnumerationType
        : kUnsignedIntegerOrEnumerationType;
    }
    static constexpr EStorageType determineStorageTypeT(double*) {
      return kDoubleType;
    }
    static constexpr EStorageType determineStorageTypeT(float*) {
      return kFloatType;
    }
    static constexpr EStorageType determineStorageTypeT(long double*) {
      return kDoubleType;
    }
    template <class T>
    static constexpr EStorageType determineStorageTypeT(T**) {
      return kPointerType;
    }
    static constexpr EStorageType determineStorageTypeT(void*) {
      return kUnsupportedType;
    }

    /// \brief Allocate storage as needed by the type.
    void ManagedAllocate();

    /// \brief Assert in case of an unsupported type. Outlined to reduce include
    ///   dependencies.
    void AssertOnUnsupportedTypeCast() const;

    size_t GetNumberOfElements() const;

    // Allow simplisticCastAs to be partially specialized.
    template<typename T>
    struct CastFwd {
      static T cast(const Value& V) {
        EStorageType storageType = V.getStorageType();
        switch (storageType) {
        case kSignedIntegerOrEnumerationType:
          return (T) V.getAs<long long>();
        case kUnsignedIntegerOrEnumerationType:
          return (T) V.getAs<unsigned long long>();
        case kDoubleType:
          return (T) V.getAs<double>();
        case kFloatType:
          return (T) V.getAs<float>();
        case kLongDoubleType:
          return (T) V.getAs<long double>();
        case kPointerType:
        case kManagedAllocation:
          return (T) (uintptr_t) V.getAs<void*>();
        case kUnsupportedType:
          V.AssertOnUnsupportedTypeCast();
        }
        return T();
      }
    };

    template<typename T>
    struct CastFwd<T*> {
      static T* cast(const Value& V) {
        EStorageType storageType = V.getStorageType();
        if (storageType == kPointerType
            || storageType == kManagedAllocation)
          return (T*) (uintptr_t) V.getAs<void*>();
        V.AssertOnUnsupportedTypeCast();
        return 0;
      }
    };

    Value(void* QualTypeAsOpaquePtr, Interpreter& Interp, EStorageType stType):
      m_StorageType(stType),
      m_Type(QualTypeAsOpaquePtr),
      m_Interpreter(&Interp) {
    }

  public:
    /// \brief Default constructor, creates a value that IsInvalid().
    Value():
      m_StorageType(kUnsupportedType), m_Type(nullptr),
      m_Interpreter(nullptr) {}
    /// \brief Copy a value.
    Value(const Value& other);
    /// \brief Move a value.
    Value(Value&& other):
      m_Storage(other.m_Storage), m_StorageType(other.m_StorageType),
      m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
      // Invalidate other so it will not release.
      other.m_StorageType = kUnsupportedType;
    }

    /// \brief Construct a valid but uninitialized Value. After this call the
    ///   value's storage can be accessed; i.e. calls ManagedAllocate() if
    ///   needed.
    Value(clang::QualType Ty, Interpreter& Interp);

    /// \brief Destruct the value; calls ManagedFree() if needed.
    ~Value();

    /// \brief Create a valid but ininitialized Value. After this call the
    ///   value's storage can be accessed; i.e. calls ManagedAllocate() if
    ///   needed.
    template <class T>
    static Value Create(void* QualTypeAsOpaquePtr, Interpreter& Interp) {
      EStorageType stType
        = std::is_reference<T>::value ?
       determineStorageTypeT((typename std::remove_reference<T>::type**)nullptr)
        : determineStorageTypeT((T*)nullptr);
      return Value(QualTypeAsOpaquePtr, Interp, stType);
    }

    Value& operator =(const Value& other);
    Value& operator =(Value&& other);

    clang::QualType getType() const;
    clang::ASTContext& getASTContext() const;
    Interpreter* getInterpreter() const { return m_Interpreter; }

    /// \brief Whether this type needs managed heap, i.e. the storage provided
    /// by the m_Storage member is insufficient, or a non-trivial destructor
    /// must be called.
    bool needsManagedAllocation() const {
      return getStorageType() == kManagedAllocation;
    }

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const;

    /// \brief Determine whether the Value is set but void.
    bool isVoid() const;

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    bool hasValue() const { return isValid() && !isVoid(); }

    /// \brief Get a reference to the value without type checking.
    /// T *must* correspond to type. Else use simplisticCastAs()!
    template <typename T>
    T& getAs() { return getAs((T*)0); }

    /// \brief Get the value without type checking.
    /// T *must* correspond to type. Else use simplisticCastAs()!
    template <typename T>
    T getAs() const { return const_cast<Value*>(this)->getAs<T>(); }

    template <typename T>
    T*& getAs(T**) { return (T*&)getAs((void**)0); }
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
    T simplisticCastAs() const {
      return CastFwd<T>::cast(*this);
    }

    ///\brief Generic interface to value printing.
    ///
    /// Can be re-implemented to print type-specific details, e.g. as
    ///\code
    ///   template <typename POSSIBLYDERIVED>
    ///   std::string printValue(const MyClass* const p, POSSIBLYDERIVED* ac,
    ///                          const Value& V);
    ///\endcode
    void print(llvm::raw_ostream& Out, bool escape = false) const;
    void dump(bool escape = true) const;
  };
} // end namespace cling

#endif // CLING_VALUE_H
