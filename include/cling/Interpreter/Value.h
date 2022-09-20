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

#include "cling/Interpreter/Visibility.h"

#include <cstdint> // for uintptr_t

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class ASTContext;
  class QualType;
}

// FIXME: Merge with clang::BuiltinType::getName
#define CLING_VALUE_BUILTIN_TYPES                                      \
  /*  X(void, Void) */                                                 \
  X(bool, Bool)                                                        \
  X(char, Char_S)                                                      \
  /*X(char, Char_U)*/                                                  \
  X(signed char, SChar)                                                \
  X(short, Short)                                                      \
  X(int, Int)                                                          \
  X(long, Long)                                                        \
  X(long long, LongLong)                                               \
  /*X(__int128, Int128)*/                                              \
  X(unsigned char, UChar)                                              \
  X(unsigned short, UShort)                                            \
  X(unsigned int, UInt)                                                \
  X(unsigned long, ULong)                                              \
  X(unsigned long long, ULongLong)                                     \
  /*X(unsigned __int128, UInt128)*/                                    \
  /*X(half, Half)*/                                                    \
  /*X(__bf16, BFloat16)*/                                              \
  X(float, Float)                                                      \
  X(double, Double)                                                    \
  X(long double, LongDouble)                                           \
  /*X(short _Accum, ShortAccum)                                        \
    X(_Accum, Accum)                                                   \
    X(long _Accum, LongAccum)                                          \
    X(unsigned short _Accum, UShortAccum)                              \
    X(unsigned _Accum, UAccum)                                         \
    X(unsigned long _Accum, ULongAccum)                                \
    X(short _Fract, ShortFract)                                        \
    X(_Fract, Fract)                                                   \
    X(long _Fract, LongFract)                                          \
    X(unsigned short _Fract, UShortFract)                              \
    X(unsigned _Fract, UFract)                                         \
    X(unsigned long _Fract, ULongFract)                                \
    X(_Sat short _Accum, SatShortAccum)                                \
    X(_Sat _Accum, SatAccum)                                           \
    X(_Sat long _Accum, SatLongAccum)                                  \
    X(_Sat unsigned short _Accum, SatUShortAccum)                      \
    X(_Sat unsigned _Accum, SatUAccum)                                 \
    X(_Sat unsigned long _Accum, SatULongAccum)                        \
    X(_Sat short _Fract, SatShortFract)                                \
    X(_Sat _Fract, SatFract)                                           \
    X(_Sat long _Fract, SatLongFract)                                  \
    X(_Sat unsigned short _Fract, SatUShortFract)                      \
    X(_Sat unsigned _Fract, SatUFract)                                 \
    X(_Sat unsigned long _Fract, SatULongFract)                        \
    X(_Float16, Float16)                                               \
    X(__float128, Float128)                                            \
    X(__ibm128, Ibm128)*/                                              \
  X(wchar_t, WChar_S)                                                  \
  /*X(wchar_t, WChar_U)*/                                              \
  /*X(char8_t, Char8)*/                                                \
  X(char16_t, Char16)                                                  \
  X(char32_t, Char32)                                                  \
  /*X(std::nullptr_t, NullPtr) same as kPtrOrObjTy*/


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
  class CLING_LIB_EXPORT Value {
  public:
    ///\brief Multi-purpose storage.
    ///
    union Storage {
#define X(type, name) type m_##name;
      CLING_VALUE_BUILTIN_TYPES
#undef X
      void* m_Ptr; /// Can point to allocation, see needsManagedAllocation().
    };

    enum TypeKind : short {
      kInvalid = 0,
#define X(type, name) \
      k##name,
      CLING_VALUE_BUILTIN_TYPES
#undef X
      kVoid,
      kPtrOrObjTy
    };

  protected:
    /// \brief The actual value.
    Storage m_Storage;

    /// \brief If the \c Value class needs to alloc and dealloc memory.
    bool m_NeedsManagedAlloc = false;

    TypeKind m_TypeKind = Value::kInvalid;

    /// \brief The value's type, stored as opaque void* to reduce
    /// dependencies.
    void* m_Type = nullptr;

    ///\brief Interpreter that produced the value.
    ///
    Interpreter* m_Interpreter = nullptr;

    /// \brief Allocate storage as needed by the type.
    void ManagedAllocate();

    /// \brief Assert in case of an unsupported type. Outlined to reduce include
    ///   dependencies.
    void AssertOnUnsupportedTypeCast() const;

    bool isPointerOrObjectType() const { return m_TypeKind == kPtrOrObjTy; }
    bool isBuiltinType() const {
      return m_TypeKind != kInvalid && !isPointerOrObjectType();
    };

    // Allow castAs to be partially specialized.
    template<typename T>
    struct CastFwd {
      static T cast(const Value& V) {
        if (V.isPointerOrObjectType())
          return (T) (uintptr_t) V.getAs<void*>();
        if (V.isInvalid() || V.isVoid()) {
#ifndef NDEBUG // Removing this might break inlining
           V.AssertOnUnsupportedTypeCast();
#endif // NDEBUG
           return T();
        }
        return V.getAs<T>();
      }
    };
    template<typename T>
    struct CastFwd<T*> {
      static T* cast(const Value& V) {
        if (V.isPointerOrObjectType())
          return (T*) (uintptr_t) V.getAs<void*>();
#ifndef NDEBUG // Removing this might break inlining
        V.AssertOnUnsupportedTypeCast();
#endif // NDEBUG
        return nullptr;
      }
    };

    /// \brief Get to the value with type checking casting the underlying
    /// stored value to T.
    template <typename T> T getAs() const {
      switch (m_TypeKind) {
      default:
#ifndef NDEBUG
        AssertOnUnsupportedTypeCast();
#endif // NDEBUG
        return T();
#define X(type, name)                                           \
        case Value::k##name: return (T) m_Storage.m_##name;
        CLING_VALUE_BUILTIN_TYPES
#undef X
      }
    }

    void AssertTypeMismatch(const char* Type) const;
  public:
    Value() = default;
    /// \brief Copy a value.
    Value(const Value& other);
    /// \brief Move a value.
    Value(Value&& other):
      m_Storage(other.m_Storage), m_NeedsManagedAlloc(other.m_NeedsManagedAlloc),
      m_TypeKind(other.m_TypeKind),
      m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
      // Invalidate other so it will not release.
      other.m_NeedsManagedAlloc = false;
      other.m_TypeKind = kInvalid;
    }

    /// \brief Construct a valid but uninitialized Value. After this call the
    ///   value's storage can be accessed; i.e. calls ManagedAllocate() if
    ///   needed.
    Value(clang::QualType Ty, Interpreter& Interp);

    /// \brief Destruct the value; calls ManagedFree() if needed.
    ~Value();

    // Avoid including type_traits.
    template<typename T>
    struct dependent_false {
       static constexpr bool value = false;
       constexpr operator bool() const noexcept { return value; }
    };
    /// \brief Create a valid Value holding a clang::Type deduced from the
    /// argument. This is useful when we want to create a \c Value with a
    /// particular value from compiled code.
    template <class T>
    static Value Create(Interpreter& Interp, T val) {
       static_assert(dependent_false<T>::value,
                     "Can not instantiate for this type.");
       return {};
    }

    Value& operator =(const Value& other);
    Value& operator =(Value&& other);

    clang::QualType getType() const;
    clang::ASTContext& getASTContext() const;
    Interpreter* getInterpreter() const { return m_Interpreter; }

    /// \brief Whether this type needs managed heap, i.e. the storage provided
    /// by the m_Storage member is insufficient, or a non-trivial destructor
    /// must be called.
    bool needsManagedAllocation() const { return m_NeedsManagedAlloc; }

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const { return m_TypeKind != Value::kInvalid; }
    bool isInvalid() const { return !isValid(); }

    /// \brief Determine whether the Value is set but void.
    bool isVoid() const { return m_TypeKind == Value::kVoid; }

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can we can get the represented value.
    bool hasValue() const { return isValid() && !isVoid(); }

    // FIXME: If the cling::Value is destroyed and it handed out an address that
    // might be accessing invalid memory.
    void** getPtrAddress() { return &m_Storage.m_Ptr; }
    void* getPtr() const { return m_Storage.m_Ptr; }
    void setPtr(void* Val) { m_Storage.m_Ptr = Val; }

#ifndef NDEBUG
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
    // FIXME: Uncomment and debug the various type mismatches.
    //#define ASSERT_TYPE_MISMATCH(name) AssertTypeMismatch(STRINGIFY(name))
    #define ASSERT_TYPE_MISMATCH(name)
#undef STRINGIFY
#undef _STRINGIFY
#else
    #define ASSERT_TYPE_MISMATCH(name)
#endif // NDEBUG
#define X(type, name)                                    \
    type get##name() const {                             \
      ASSERT_TYPE_MISMATCH(name);                        \
      return m_Storage.m_##name;                         \
    }                                                    \
    void set##name(type Val) {                           \
      ASSERT_TYPE_MISMATCH(name);                        \
      m_Storage.m_##name = Val;                          \
    }                                                    \

  CLING_VALUE_BUILTIN_TYPES

#undef X

    /// \brief Get the value with cast.
    //
    /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
    /// casting the value of builtins (except void), enums and pointers.
    /// Values referencing an object are treated as pointers to the object.
    template <typename T>
    T castAs() const {
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

  template <> inline void* Value::getAs() const {
    if (isPointerOrObjectType())
      return m_Storage.m_Ptr;
    return (void*)getAs<uintptr_t>();
  }

#define X(type, name)                                                   \
  template <> Value Value::Create(Interpreter& Interp, type val);       \

  CLING_VALUE_BUILTIN_TYPES

#undef X

} // end namespace cling

#endif // CLING_VALUE_H
