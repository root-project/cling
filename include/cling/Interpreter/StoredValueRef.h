//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_STOREDVALUEREF_H
#define CLING_STOREDVALUEREF_H

#include "cling/Interpreter/Value.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace cling {
  ///\brief A type, value pair.
  //
  /// Reference counted wrapper around a cling::Value with storage allocation.
  class StoredValueRef {
  private:
    class StoredValue: public Value, public llvm::RefCountedBase<StoredValue> {
    public:
      /// \brief Construct a valid StoredValue, allocating as needed.
      StoredValue(const clang::ASTContext&, clang::QualType t);
      /// \brief Destruct and deallocate if necessary.
      ~StoredValue();

      /// \brief Number of bytes that need to be allocated to hold a value
      /// of our type
      int64_t getAllocSizeInBytes(const clang::ASTContext& ctx) const;

      /// \brief Take ownership of an existing char[]
      void adopt(char* addr);

      /// \brief Memory allocated for value, owned by this value
      ///
      /// Points to memory allocated for storing the value, if it
      /// does not fit into Value::value.
      char* Mem;
    };

    llvm::IntrusiveRefCntPtr<StoredValue> fValue;

    StoredValueRef(StoredValue* value): fValue(value) {}

  public:
    /// \brief Allocate an object of type t and return a StoredValueRef to it.
    static StoredValueRef allocate(const clang::ASTContext& ctx,
                                   clang::QualType t);
    /// \brief Create a bitwise copy of value wrapped in a StoredValueRef.
    static StoredValueRef bitwiseCopy(const clang::ASTContext& ctx,
                                      const Value& value);
    /// \brief Create a bitwise copy of svalue.
    static StoredValueRef bitwiseCopy(const clang::ASTContext& ctx,
                                      const StoredValueRef svalue) {
      return bitwiseCopy(ctx, *svalue.fValue);
    }

    static StoredValueRef invalidValue() { return StoredValueRef(); }

    /// \brief Construct an empty, invalid value.
    StoredValueRef() {}

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const { return fValue; }

    /// \brief Determine whether the Value needs to manage an allocation.
    bool needsManagedAllocation() const { return fValue->Mem; }

    /// \brief Determine whether the Value is set but void.
    /// For compatibility with cling::Value.
    bool isVoid(const clang::ASTContext&) const { return isVoid(); }
    /// \brief Determine whether the Value is set but void.
    /// For compatibility with cling::Value.
    bool isVoid() const { return false; }

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    /// For compatibility with cling::Value.
    bool hasValue(const clang::ASTContext& ASTContext) const {
      return hasValue(); }
    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    /// For compatibility with cling::Value.
    bool hasValue() const { return isValid(); }

    const Value& get() const { return *fValue; }
  };
} // end namespace cling

#endif // CLING_STOREDVALUEREF_H
