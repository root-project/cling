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
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <sstream>

namespace {

  ///\brief The allocation starts with this layout; it is followed by the
  ///  value's object at m_Payload. This class does not inherit from
  ///  llvm::RefCountedBase because deallocation cannot use this type but must
  ///  free the character array.

  class AllocatedValue {
  public:
    typedef void (*DtorFunc_t)(void*);

  private:
    ///\brief The reference count - once 0, this object will be deallocated.
    mutable unsigned m_RefCnt;

    ///\brief The destructor function.
    DtorFunc_t m_DtorFunc;

    ///\brief The size of the allocation (for arrays)
    unsigned long m_AllocSize;

    ///\brief The number of elements in the array
    unsigned long m_NElements;

    ///\brief The start of the allocation.
    char m_Payload[1];

    static DtorFunc_t PtrToFunc(void* ptr) {
      union {
        void* m_Ptr;
        DtorFunc_t m_Func;
      };
      m_Ptr = ptr;
      return m_Func;
    }


  public:
    ///\brief Initialize the storage management part of the allocated object.
    ///  The allocator is referencing it, thus initialize m_RefCnt with 1.
    ///\param [in] dtorFunc - the function to be called before deallocation.
    AllocatedValue(void* dtorFunc, size_t allocSize, size_t nElements):
      m_RefCnt(1), m_DtorFunc(PtrToFunc(dtorFunc)), m_AllocSize(allocSize),
      m_NElements(nElements)
    {}

    char* getPayload() { return m_Payload; }

    static unsigned getPayloadOffset() {
      static const AllocatedValue Dummy(0,0,0);
      return Dummy.m_Payload - (const char*)&Dummy;
    }

    static AllocatedValue* getFromPayload(void* payload) {
      return
        reinterpret_cast<AllocatedValue*>((char*)payload - getPayloadOffset());
    }

    void Retain() { ++m_RefCnt; }

    ///\brief This object must be allocated as a char array. Deallocate it as
    ///   such.
    void Release() {
      assert (m_RefCnt > 0 && "Reference count is already zero.");
      if (--m_RefCnt == 0) {
        if (m_DtorFunc) {
          char* payload = getPayload();
          for (size_t el = 0; el < m_NElements; ++el)
            (*m_DtorFunc)(payload + el * m_AllocSize / m_NElements);
        }
        delete [] (char*)this;
      }
    }
  };
}

namespace cling {

  Value::Value(const Value& other):
    m_Storage(other.m_Storage), m_StorageType(other.m_StorageType),
    m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
    if (other.needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Retain();
  }

  Value::Value(Value&& other):
    m_Storage(other.m_Storage), m_StorageType(other.m_StorageType),
    m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
    // Invalidate other so it will not release.
    other.m_StorageType = kUnsupportedType;
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
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();

    // Retain new one.
    m_Type = other.m_Type;
    m_Storage = other.m_Storage;
    m_StorageType = other.m_StorageType;
    m_Interpreter = other.m_Interpreter;
    if (needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Retain();
    return *this;
  }

  Value& Value::operator =(Value&& other) {
    // Release old value.
    if (needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();

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
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();
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

  unsigned long Value::GetNumberOfElements() const {
    if (const clang::ConstantArrayType* ArrTy
        = llvm::dyn_cast<clang::ConstantArrayType>(getType())) {
      llvm::APInt arrSize(sizeof(unsigned long)*8, 1);
      do {
        arrSize *= ArrTy->getSize();
        ArrTy = llvm::dyn_cast<clang::ConstantArrayType>(ArrTy->getElementType()
                                                         .getTypePtr());
      } while (ArrTy);
      return (unsigned long)arrSize.getZExtValue();
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
    void* dtorFunc = 0;
    clang::QualType DtorType = getType();
    // For arrays we destruct the elements.
    if (const clang::ConstantArrayType* ArrTy
        = llvm::dyn_cast<clang::ConstantArrayType>(DtorType.getTypePtr())) {
      DtorType = ArrTy->getElementType();
    }
    if (const clang::RecordType* RTy = DtorType->getAs<clang::RecordType>())
      dtorFunc = GetDtorWrapperPtr(RTy->getDecl());

    const clang::ASTContext& ctx = getASTContext();
    unsigned payloadSize = ctx.getTypeSizeInChars(getType()).getQuantity();
    char* alloc = new char[AllocatedValue::getPayloadOffset() + payloadSize];
    AllocatedValue* allocVal = new (alloc) AllocatedValue(dtorFunc, payloadSize,
                                                          GetNumberOfElements());
    m_Storage.m_Ptr = allocVal->getPayload();
  }

  void Value::AssertOnUnsupportedTypeCast() const {
    assert("unsupported type in Value, cannot cast simplistically!" && 0);
  }

  /// \brief Get the function address of the wrapper of the destructor.
  void* Value::GetDtorWrapperPtr(const clang::RecordDecl* RD) const {
    std::string funcname;
    {
      llvm::raw_string_ostream namestr(funcname);
      namestr << "__cling_StoredValue_Destruct_" << RD;
    }

    // Check whether the function exists before calling
    // utils::TypeName::GetFullyQualifiedName which is expensive
    // (memory-wise). See ROOT-6909.
    std::string code;
    if (!m_Interpreter->getAddressOfGlobal(funcname)) {
      code = "extern \"C\" void ";
      clang::QualType RDQT(RD->getTypeForDecl(), 0);
      std::string typeName
        = utils::TypeName::GetFullyQualifiedName(RDQT, RD->getASTContext());
      std::string dtorName = RD->getNameAsString();
      code += funcname + "(void* obj){((" + typeName + "*)obj)->~"
        + dtorName + "();}";
    }
    // else we have an empty code string - but the function alreday exists
    // so we'll be fine and take the existing one (ifUniq = true).

    return m_Interpreter->compileFunction(funcname, code, true /*ifUniq*/,
                                          false /*withAccessControl*/);
  }

  namespace valuePrinterInternal {
    std::string printTypeInternal(const Value& V);
    std::string printValueInternal(const Value& V);
  } // end namespace valuePrinterInternal

  void Value::print(llvm::raw_ostream& Out) const {

    // Get the default type string representation
    std::string typeStr = cling::valuePrinterInternal::printTypeInternal(*this);
    // Get the value string representation, by printValue() method overloading
    std::string valueStr = cling::valuePrinterInternal::printValueInternal(*this);

    // Print the type and the value:
    Out << typeStr + " " + valueStr << "\n";
  }

  void Value::dump() const {
    // We need stream that doesn't close its file descriptor, thus we are not
    // using llvm::outs. Keeping file descriptor open we will be able to use
    // the results in pipes (Savannah #99234).

    // Alternatively we could use llvm::errs()
    std::unique_ptr<llvm::raw_ostream> Out;
    Out.reset(new llvm::raw_os_ostream(std::cout));
    print(*Out.get());
  }
} // end namespace cling
