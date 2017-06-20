//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Casting.h"
#include "cling/Utils/Output.h"
#include "cling/Utils/UTF8.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/CanonicalType.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_os_ostream.h"

namespace {

  ///\brief The allocation starts with this layout; it is followed by the
  ///  value's object at m_Payload. This class does not inherit from
  ///  llvm::RefCountedBase because deallocation cannot use this type but must
  ///  free the character array.

  class AllocatedValue {
  public:
    typedef void (*DtorFunc_t)(void*);

  private:
    ///\brief The destructor function.
    DtorFunc_t m_DtorFunc;

    ///\brief The size of the allocation (for arrays)
    size_t m_AllocSize;

    ///\brief The number of elements in the array
    size_t m_NElements;

    ///\brief The reference count - once 0, this object will be deallocated.
    /// Hopefully 2^55 - 1 references should be enough as the last byte is
    /// used for flag storage.
    enum { SizeBytes = sizeof(size_t), ConstructedByte = SizeBytes - 1 };
    union {
      size_t m_Count;
      char   m_Bytes[SizeBytes];
    };

    size_t UpdateRefCount(int Amount) {
      // Bit shift the bytes used in m_Bytes for representing an integer
      // respecting endian-ness and which of those bytes are significant.
      assert((Amount == 1 || Amount == -1) && "Invalid amount");
      union { size_t m_Count; char m_Bytes[SizeBytes]; } RC = { 0 };
      const size_t NBytes = SizeBytes - sizeof(char);
      const size_t Endian = llvm::sys::IsBigEndianHost;
      ::memcpy(&RC.m_Bytes[Endian], &m_Bytes[0], NBytes);
      RC.m_Count += Amount;
      ::memcpy(&m_Bytes[0], &RC.m_Bytes[Endian], NBytes);
      return RC.m_Count;
    }

    static AllocatedValue* FromPtr(void* Ptr) {
      return reinterpret_cast<AllocatedValue*>(reinterpret_cast<char*>(Ptr) -
                                               sizeof(AllocatedValue));
    }

    ///\brief Initialize the storage management part of the allocated object.
    ///  The allocator is referencing it, thus initialize m_RefCnt with 1.
    ///\param [in] dtorFunc - the function to be called before deallocation.
    AllocatedValue(size_t Size, size_t NElem, DtorFunc_t Dtor) :
      m_DtorFunc(Dtor), m_AllocSize(Size), m_NElements(NElem) {
      m_Count = 0;
      UpdateRefCount(1);
    }

  public:
    ///\brief Create an AllocatedValue whose lifetime is reference counted.
    /// \returns The address of the writeable client data.
    static void* Create(size_t Size, size_t NElem, DtorFunc_t Dtor) {
      char* Alloc = new char[sizeof(AllocatedValue) + Size];
      AllocatedValue* AV = new (Alloc) AllocatedValue(Size, NElem, Dtor);

      static_assert(std::is_standard_layout<AllocatedValue>::value, "padding");
      static_assert(sizeof(m_Count) == sizeof(m_Bytes), "union padding");
      static_assert(((offsetof(AllocatedValue, m_Count) + sizeof(m_Count)) %
                            SizeBytes) == 0,
                    "Buffer may not be machine aligned");
      assert(&Alloc[sizeof(AllocatedValue) - 1] == &AV->m_Bytes[SizeBytes - 1]
             && "Padded AllocatedValue");

      // Give back the first client writable byte.
      return AV->m_Bytes + SizeBytes;
    }

    static void Retain(void* Ptr) {
      FromPtr(Ptr)->UpdateRefCount(1);
    }

    ///\brief This object must be allocated as a char array. Deallocate it as
    ///   such.
    static void Release(void* Ptr) {
      AllocatedValue* AV = FromPtr(Ptr);
      if (AV->UpdateRefCount(-1) == 0) {
        if (AV->m_DtorFunc && AV->m_Bytes[ConstructedByte]) {
          assert(AV->m_NElements && "No elements!");
          char* Payload = reinterpret_cast<char*>(Ptr);
          const auto Skip = AV->m_AllocSize / AV->m_NElements;
          while (AV->m_NElements-- != 0)
            (*AV->m_DtorFunc)(Payload + AV->m_NElements * Skip);
        }
        this->~AllocatedValue();
        delete [] reinterpret_cast<char*>(AV);
      }
    }
  };
}

namespace cling {

  Value::Value(const Value& other):
    m_Storage(other.m_Storage), m_StorageType(other.m_StorageType),
    m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
    if (other.needsManagedAllocation())
      AllocatedValue::Retain(m_Storage.m_Ptr);
  }

  Value::Value(clang::QualType clangTy, Interpreter& Interp):
    m_StorageType(determineStorageType(clangTy)),
    m_Type(clangTy.getAsOpaquePtr()),
    m_Interpreter(&Interp) {
    if (needsManagedAllocation())
      ManagedAllocate();
  }

  Value& Value::operator =(const Value& other) {
    // Release old value.
    if (needsManagedAllocation())
      AllocatedValue::Release(m_Storage.m_Ptr);

    // Retain new one.
    m_Type = other.m_Type;
    m_Storage = other.m_Storage;
    m_StorageType = other.m_StorageType;
    m_Interpreter = other.m_Interpreter;
    if (needsManagedAllocation())
      AllocatedValue::Retain(m_Storage.m_Ptr);
    return *this;
  }

  Value& Value::operator =(Value&& other) {
    // Release old value.
    if (needsManagedAllocation())
      AllocatedValue::Release(m_Storage.m_Ptr);

    // Move new one.
    m_Type = other.m_Type;
    m_Storage = other.m_Storage;
    m_StorageType = other.m_StorageType;
    m_Interpreter = other.m_Interpreter;
    // Invalidate other so it will not release.
    other.m_StorageType = kUnsupportedType;

    return *this;
  }

  Value::~Value() {
    if (needsManagedAllocation())
      AllocatedValue::Release(m_Storage.m_Ptr);
  }

  clang::QualType Value::getType() const {
    return clang::QualType::getFromOpaquePtr(m_Type);
  }

  clang::ASTContext& Value::getASTContext() const {
    return m_Interpreter->getCI()->getASTContext();
  }

  bool Value::isValid() const { return !getType().isNull(); }

  bool Value::isVoid() const {
    const clang::ASTContext& Ctx = getASTContext();
    return isValid() && Ctx.hasSameType(getType(), Ctx.VoidTy);
  }

  size_t Value::GetNumberOfElements() const {
    if (const clang::ConstantArrayType* ArrTy
        = llvm::dyn_cast<clang::ConstantArrayType>(getType())) {
      llvm::APInt arrSize(sizeof(size_t)*8, 1);
      do {
        arrSize *= ArrTy->getSize();
        ArrTy = llvm::dyn_cast<clang::ConstantArrayType>(ArrTy->getElementType()
                                                         .getTypePtr());
      } while (ArrTy);
      return static_cast<size_t>(arrSize.getZExtValue());
    }
    return 1;
  }

  Value::EStorageType Value::determineStorageType(clang::QualType QT) {
    const clang::Type* desugCanon = QT.getCanonicalType().getTypePtr();
    if (desugCanon->isSignedIntegerOrEnumerationType())
      return kSignedIntegerOrEnumerationType;
    else if (desugCanon->isUnsignedIntegerOrEnumerationType())
      return kUnsignedIntegerOrEnumerationType;
    else if (desugCanon->isRealFloatingType()) {
      const clang::BuiltinType* BT = desugCanon->getAs<clang::BuiltinType>();
      if (BT->getKind() == clang::BuiltinType::Double)
        return kDoubleType;
      else if (BT->getKind() == clang::BuiltinType::Float)
        return kFloatType;
      else if (BT->getKind() == clang::BuiltinType::LongDouble)
        return kLongDoubleType;
    } else if (desugCanon->isPointerType() || desugCanon->isObjectType()
               || desugCanon->isReferenceType()) {
      if (desugCanon->isRecordType() || desugCanon->isConstantArrayType()
          || desugCanon->isMemberPointerType())
        return kManagedAllocation;
      return kPointerType;
    }
    return kUnsupportedType;
  }

  void Value::ManagedAllocate() {
    assert(needsManagedAllocation() && "Does not need managed allocation");
    const clang::QualType Ty = getType();
    clang::QualType DtorTy = Ty;

    // For arrays we destruct the elements.
    if (const clang::ConstantArrayType* ArrTy =
            llvm::dyn_cast<clang::ConstantArrayType>(Ty.getTypePtr())) {
      DtorTy = ArrTy->getElementType();
    }

    AllocatedValue::DtorFunc_t DtorFunc = nullptr;
    if (const clang::RecordType* RTy = DtorTy->getAs<clang::RecordType>()) {
      DtorFunc = cling::utils::VoidToFunctionPtr<AllocatedValue::DtorFunc_t>(
          m_Interpreter->compileDtorCallFor(RTy->getDecl()));
    }

    m_Storage.m_Ptr = AllocatedValue::Create(
        getASTContext().getTypeSizeInChars(Ty).getQuantity(),
        GetNumberOfElements(), DtorFunc);
  }

  void Value::AssertOnUnsupportedTypeCast() const {
    assert("unsupported type in Value, cannot cast simplistically!" && 0);
  }

  namespace valuePrinterInternal {
    std::string printTypeInternal(const Value& V);
    std::string printValueInternal(const Value& V);
  } // end namespace valuePrinterInternal

  void Value::print(llvm::raw_ostream& Out, bool Escape) const {
    // Save the default type string representation so output can occur as one
    // operation (calling printValueInternal below may write to stderr).
    const std::string Type = valuePrinterInternal::printTypeInternal(*this);

    // Get the value string representation, by printValue() method overloading
    const std::string Val = cling::valuePrinterInternal::printValueInternal(*this);
    if (Escape) {
      const char* Data = Val.data();
      const size_t N = Val.size();
      switch (N ? Data[0] : 0) {
        case 'u': case 'U': case 'L':
          if (N < 3 || Data[1] != '\"')
            break;

          // Unicode string, encoded as Utf-8
        case '\"':
          if (N > 2 && Data[N-1] == '\"') {
            // Drop the terminating " so Utf-8 errors can be detected ("\xeA")
            Out << Type << ' ';
            utils::utf8::EscapeSequence().encode(Data, N-1, Out) << "\"\n";
            return;
          }
        default:
          break;
      }
    }
    Out << Type << ' ' << Val << '\n';
  }

  void Value::dump(bool Escape) const {
    print(cling::outs(), Escape);
  }
} // end namespace cling
