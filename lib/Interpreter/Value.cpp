//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "llvm/Support/raw_ostream.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/CanonicalType.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"

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

    ///\brief The start of the allocation.
    char m_Payload[1];

  public:
    ///\brief Initialize the storage management part of the allocated object.
    ///  The allocator is referencing it, thus initialize m_RefCnt with 1.
    ///\param [in] dtorFunc - the function to be called before deallocation.
    AllocatedValue(void* dtorFunc):
      m_RefCnt(1), m_DtorFunc((DtorFunc_t)dtorFunc) {}

    char* getPayload() { return m_Payload; }

    static unsigned getPayloadOffset() {
      static const AllocatedValue Dummy(0);
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
        if (m_DtorFunc)
          (*m_DtorFunc)(getPayload());
        delete [] (char*)this;
      }
    }
  };
}

namespace cling {

Value::Value(const Value& other) : m_Type(other.m_Type) {
  if (needsManagedAllocation())
    AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Retain();
}

Value::Value(clang::QualType clangTy, Interpreter* Interp):
  m_Type(clangTy.getAsOpaquePtr()) {
  if (needsManagedAllocation())
    ManagedAllocate(Interp);
}

Value& Value::operator =(const Value& other) {
  m_Type = other.m_Type;
  if (needsManagedAllocation())
    AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Retain();
  return *this;
}

Value::~Value() {
  if (needsManagedAllocation())
    AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();
}

clang::QualType Value::getType() const {
  return clang::QualType::getFromOpaquePtr(m_Type);
}

bool Value::isValid() const { return !getType().isNull(); }

bool Value::isVoid(const clang::ASTContext& ASTContext) const {
  return isValid() && ASTContext.hasSameType(getType(), ASTContext.VoidTy);
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
  return !getType()->getUnqualifiedDesugaredType()->isBuiltinType();
}

void Value::ManagedAllocate(Interpreter* interp) {
  assert(interp && "This type requires the interpreter for value allocation");
  void* dtorFunc = 0;
  if (const clang::RecordType* RTy
      = clang::dyn_cast<clang::RecordType>(getType()))
    dtorFunc = GetDtorWrapperPtr(RTy->getDecl(), *interp);

  const clang::ASTContext& ctx = interp->getCI()->getASTContext();
  unsigned payloadSize = ctx.getTypeSizeInChars(getType()).getQuantity();
  char* alloc = new char[AllocatedValue::getPayloadOffset() + payloadSize];
  AllocatedValue* allocVal = new (alloc) AllocatedValue(dtorFunc);
  m_Storage.m_Ptr = allocVal->getPayload();
}

void Value::AssertOnUnsupportedTypeCast() const {
  assert("unsupported type in Value, cannot cast simplistically!" && 0);
}

/// \brief Get the function address of the wrapper of the destructor.
void* Value::GetDtorWrapperPtr(const clang::RecordDecl* RD,
                               Interpreter& interp) const {
  std::string funcname;
  {
    llvm::raw_string_ostream namestr(funcname);
    namestr << "__cling_StoredValue_Destruct_" << RD;
  }

  std::string code("extern \"C\" void ");
  {
    std::string typeName
      = utils::TypeName::GetFullyQualifiedName(getType(),
                                               RD->getASTContext());
    std::string dtorName = RD->getNameAsString();
    code += funcname + "(void* obj){((" + typeName + "*)obj)->~"
      + dtorName + "();}";
  }

  return interp.compileFunction(funcname, code, true /*ifUniq*/,
                                false /*withAccessControl*/);
}
} // namespace cling
