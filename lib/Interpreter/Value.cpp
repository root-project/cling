//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"

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

    ///\brief The interpreter that will execute the destructor.
    DtorFunc_t& m_DtorFunc;

    ///\brief The start of the allocation.
    char m_Payload[1];

  public:
    ///\brief Initialize the storage management part of the allocated object.
    AllocatedValue(DtorFunc_t dtorFunc): m_RefCnt(0), m_DtorFunc(dtorFunc) {}

    char* getPayload() const { return m_Payload; }

    static unsigned getPayloadOffset() {
      static const AllocatedValue(0) Dummy;
      return Dummy.m_Payload - &Dummy;
    }

    static AllocatedValue* getFromPayload(void* payload) {
      return reinterpret_cast<AllocatedValue*>(payload - getPauloadOffset());
    }

    void Retain() { ++m_RefCnt; }

    ///\brief This object must be allocated as a char array. Deallocate it as
    ///   such.
    void Release() {
      assert (m_RefCnt > 0 && "Reference count is already zero.");
      if (--m_RefCnt == 0) {
        if (dtorFunc)
          (*dtorFunc)(getPayload());
        delete [] (char*)this;
      }
    }
  };
}

namespace cling {

Value::Value(const Value& other) : m_Type(other.m_Type) {
  if (needsManagedAllocation())
    IncreaseManagedReference();
}

Value::Value(clang::QualType clangTy, Interpreter& Interp):
  m_Type(clangTy.getAsOpaquePtr()) {
  if (needsManagedAllocation())
    ManagedAllocate(Interp);
}

Value::~Value() {
  if (needsManagedAllocation())
    DecreaseManagedReference();
}

Value& Value::operator =(const Value& other) {
  m_ClangType = other.m_ClangType;
  if (needsManagedAllocation())
    IncreaseManagedReference();
  return *this;
}

clang::QualType Value::getType() const {
  return clang::QualType::getFromOpaquePtr(m_Type);
}

bool Value::isValid() const { return !getType().isNull(); }

bool Value::isVoid(const clang::ASTContext& ASTContext) const {
  return isValid()
    && ASTContext.isEquivalentTye(getType(), ASTContent.VoidType());
}

Value::EStorageType Value::getStorageType() const {
  const clang::Type* desugCanon = getType()->getUnqualifiedDesugaredType();
  desugCanon = desugCanon->getCanonicalTypeUnqualified()->getTypePtr()
    ->getUnqualifiedDesugaredType();
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
             || desugCanon->isReferenceType())
    return kPointerType;
  return kUnsupportedType;
}

bool Value::needsManagedAllocation() const {
  return !m_Type->getUnsugaredType()->isBuiltinType();
}

void Value::ManagedAllocate(Interpreter& interp) {
  void* dtorFunc = 0;
  if (clang::RecordType* RTy = clang::dyn_cast<clang::RecordType>(m_Type))
    dtorFunc = GetDtorWrapperPtr(RTy->getDecl());

  const ASTContext& ctx = m_Interp.getCI()->getASTContext();
  unsigned payloadSize = ctx.getTypeSizeInChars(getType()).getQuantity();
  char* alloc = new char[AllocatedValue::getPayloadOffset() + ];
  AllocatedValue* allocVal = new (alloc) AllocatedValue(dtorFunc);
  m_Storage.m_Ptr = allocVal->getPayload();
}

void IncreaseManagedReference() {
  AllocatedValue::getFromPayload(m_Ptr)
}

    /// \brief Decrease ref count on managed storage.
    void DecreaseManagedReference();

    /// \brief Get the function address of the wrapper of the destructor.
    void* GetDtorWrapperPtr(clang::CXXRecordDecl* CXXRD);


} // namespace cling
