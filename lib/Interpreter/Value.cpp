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

  ///\brief The layout/usage of memory allocated by AllocatedValue::Create
  /// is dependent on the the type of object it is representing.  If the type
  /// has a non-trival destructor then the memory base will point to either
  /// a full Destructable struct (when it is also an array whose size > 1), or
  /// a single DtorFunc_t value when the object is a single instance or an array
  /// with only 1 element.
  ///
  /// If neither of these are true (a POD or array to one), or directly follwing
  /// the prior two cases, is the memory location that can be used to placement
  /// new an AllocatedValue instance.
  ///
  /// The AllocatedValue instance ontains a single union for reference counting
  /// and flags of how the layout exists in memory.
  ///
  /// On 32-bit the reference count will max out a bit before 16.8 million.
  /// 64 bit limit is still extremely high (2^56)-1
  ///
  ///
  /// General layout of memory allocated by AllocatedValue::Create
  ///
  /// +---- Destructable ---+  <- Optional, allocated for arrays
  /// |                     |      TestFlag(kHasElements)
  /// |  Size               |
  /// |  Elements           |
  /// |  Dtor               |  <- Can exist without prior Destructable members
  /// |                     |      TestFlag(kHasDestructor)
  /// |                     |
  /// +---AllocatedValue ---+  <- This object
  /// |                     |      TestFlag(kHasElements)
  /// | union {             |
  /// |  size_t m_Count     |
  /// |  char   m_Bytes[8]  |  <- m_Bytes[7] is reserved for AllocatedValue
  /// | };                  |     & ValueExtractionSynthesizer writing info.
  /// |                     |
  /// +~~ Client Payload ~~~+  <- Returned from AllocatedValue::Create
  /// |                     |
  /// |                     |
  /// +---------------------+
  ///
  /// It may be possible to ignore the caching of this info all together and
  /// just figure out what to do in AllocatedValue::Release by passing the
  /// QualType and Interpreter, but am a bit weary of this for two reasons:
  ///
  ///  1. FIXME: There is still a bad lifetime cycle where a Value referencing
  ///     an Interpreter that has been destroyed is possible.
  ///  2. How that might interact with decl unloading, and the possibility of
  ///     a destructor no longer being defined after a cling::Value has been
  ///     created to represent a fuller state of the type.

  class AllocatedValue {
  public:
    typedef void (*DtorFunc_t)(void*);

  private:

    struct Destructable {
      ///\brief Size to skip to get the next element in the array.
      size_t Size;

      ///\brief Total number of elements in the array.
      size_t Elements;

      ///\brief The destructor function.
      DtorFunc_t Dtor;
    };

    ///\brief The reference count - once 0, this object will be deallocated.
    /// Hopefully 2^55 - 1 references should be enough as the last byte is
    /// used for flag storage.
    enum {
      SizeBytes = sizeof(size_t),
      FlagsByte = SizeBytes - 1,
    
      kConstructorRan = 1, // Used by ValueExtractionSynthesizer
      kHasDestructor = 2,
      kHasElements = 4
    };
    union {
      size_t m_Count;
      char   m_Bytes[SizeBytes];
    };

    bool TestFlags(unsigned F) const { return (m_Bytes[FlagsByte] & F) == F; }

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

    template <class T = AllocatedValue> static T* FromPtr(void* Ptr) {
      return reinterpret_cast<T*>(reinterpret_cast<char*>(Ptr) - sizeof(T));
    }

    ///\brief Initialize the reference count and flag management.
    /// Everything else is in a Destructable object before -this-
    AllocatedValue(char Info) {
      m_Count = 0;
      m_Bytes[FlagsByte] = Info;
      UpdateRefCount(1);
    }

  public:

    ///\brief Create an AllocatedValue.
    /// \returns The address of the writeable client data.
    static void* Create(size_t Size, size_t NElem, DtorFunc_t Dtor) {
      size_t AllocSize = sizeof(AllocatedValue) + Size;
      size_t ExtraSize = 0;
      char Flags = 0;
      if (Dtor) {
        // Only need the elements data for arrays larger than 1.
        if (NElem > 1) {
          Flags |= kHasElements;
          ExtraSize = sizeof(Destructable);
        } else
          ExtraSize = sizeof(DtorFunc_t);

        Flags |= kHasDestructor;
        AllocSize += ExtraSize;
      }

      char* Alloc = new char[AllocSize];

      if (Dtor) {
        // Move the Buffer ptr to where AllocatedValue begins
        Alloc += ExtraSize;
        // Now back up to get the location of the Destructable members
        // This is so writing to Destructable::Dtor will work when only
        // additional space for DtorFunc_t was written.
        Destructable* DS = FromPtr<Destructable>(Alloc);
        if (NElem > 1) {
          DS->Elements = NElem;
          // Hopefully there won't be any issues with object alignemnt of arrays
          // If there are, that would have to be dealt with here and write the
          // proper skip amount in DS->Size.
          DS->Size = Size / NElem;
        }
        DS->Dtor = Dtor;
      }

      AllocatedValue* AV = new (Alloc) AllocatedValue(Flags);

      // Just make sure alignment is as expected.
      static_assert(std::is_standard_layout<Destructable>::value, "padding");
      static_assert((sizeof(Destructable) % SizeBytes) == 0, "alignment");
      static_assert(std::is_standard_layout<AllocatedValue>::value, "padding");
      static_assert(sizeof(m_Count) == sizeof(m_Bytes), "union padding");
      static_assert(((offsetof(AllocatedValue, m_Count) + sizeof(m_Count)) %
                            SizeBytes) == 0,
                    "Buffer may not be machine aligned");
      // Validate the byte ValueExtractionSynthesizer will write too
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
        if (AV->TestFlags(kConstructorRan|kHasDestructor)) {
          Destructable* Dtor = FromPtr<Destructable>(AV);
          size_t Elements = 1, Size = 0;
          if (AV->TestFlags(kHasElements)) {
            Elements = Dtor->Elements;
            Size = Dtor->Size;
          }
          char* Payload = reinterpret_cast<char*>(Ptr);
          while (Elements-- != 0)
            (*Dtor->Dtor)(Payload + Elements * Size);
        }

        // Subtract the amount that was over-allocated from the base of -this-
        char* Allocated = reinterpret_cast<char*>(AV);
        if (AV->TestFlags(kHasElements))
          Allocated -= sizeof(Destructable);
        else if (AV->TestFlags(kHasDestructor))
          Allocated -= sizeof(DtorFunc_t);

        AV->~AllocatedValue();
        delete [] Allocated;
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
