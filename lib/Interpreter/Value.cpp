//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: Value.cpp 48537 2013-02-11 17:30:03Z vvassilev $
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "llvm/ExecutionEngine/GenericValue.h"

#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"

namespace cling {

Value::Value():
  m_ClangType(),
  m_LLVMType()
{
  assert(sizeof(llvm::GenericValue) <= sizeof(m_GV)
         && "GlobalValue buffer too small");
  new (m_GV) llvm::GenericValue();
}

Value::Value(const Value& other):
  m_ClangType(other.m_ClangType),
  m_LLVMType(other.m_LLVMType)
{
  assert(sizeof(llvm::GenericValue) <= sizeof(m_GV)
         && "GlobalValue buffer too small");
  new (m_GV) llvm::GenericValue(other.getGV());
}

Value::Value(const llvm::GenericValue& v, clang::QualType t) 
  : m_ClangType(t.getAsOpaquePtr()), m_LLVMType(0)
{
  assert(sizeof(llvm::GenericValue) <= sizeof(m_GV)
         && "GlobalValue buffer too small");
  new (m_GV) llvm::GenericValue(v);
}

Value::Value(const llvm::GenericValue& v, clang::QualType clangTy, 
          const llvm::Type* llvmTy) 
  : m_ClangType(clangTy.getAsOpaquePtr()), m_LLVMType(llvmTy)
{
  assert(sizeof(llvm::GenericValue) <= sizeof(m_GV)
         && "GlobalValue buffer too small");
  new (m_GV) llvm::GenericValue(v);
}

Value& Value::operator =(const Value& other) {
  m_ClangType = other.m_ClangType;
  m_LLVMType = other.m_LLVMType;
  setGV(other.getGV());
  return *this;
}

llvm::GenericValue Value::getGV() const {
  return reinterpret_cast<const llvm::GenericValue&>(m_GV);
}
void Value::setGV(llvm::GenericValue GV) {
  assert(sizeof(llvm::GenericValue) <= sizeof(m_GV)
         && "GlobalValue buffer too small");
  reinterpret_cast<llvm::GenericValue&>(m_GV) = GV;
}

clang::QualType Value::getClangType() const {
  return clang::QualType::getFromOpaquePtr(m_ClangType);
}

bool Value::isValid() const { return !getClangType().isNull(); }

bool Value::isVoid(const clang::ASTContext& ASTContext) const {
  return isValid() 
    && getClangType().getDesugaredType(ASTContext)->isVoidType();
}

Value::EStorageType Value::getStorageType() const {
  const clang::Type* desugCanon = getClangType()->getUnqualifiedDesugaredType();
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
  } else if (desugCanon->isPointerType() || desugCanon->isObjectType() || desugCanon->isReferenceType())
    return kPointerType;
  return kUnsupportedType;
}

void* Value::getAs(void**) const { return getGV().PointerVal; }
double Value::getAs(double*) const { return getGV().DoubleVal; }
long double Value::getAs(long double*) const {
  return getAs((double*)0);
}
float Value::getAs(float*) const { return getGV().FloatVal; }
bool Value::getAs(bool*) const { return getGV().IntVal.getBoolValue(); }
signed char Value::getAs(signed char*) const {
  return (signed char) getAs((signed long long*)0);
}
unsigned char Value::getAs(unsigned char*) const {
  return (unsigned char) getAs((unsigned long long*)0);
}
signed short Value::getAs(signed short*) const {
  return (signed short) getAs((signed long long*)0);
}
unsigned short Value::getAs(unsigned short*) const {
  return (unsigned short) getAs((unsigned long long*)0);
}
signed int Value::getAs(signed int*) const {
  return (signed int) getAs((signed long long*)0);
}
unsigned int Value::getAs(unsigned int*) const {
  return (unsigned int) getAs((unsigned long long*)0);
}
signed long Value::getAs(signed long*) const {
  return (long) getAs((signed long long*)0);
}
unsigned long Value::getAs(unsigned long*) const {
  return (signed long) getAs((unsigned long long*)0);
}
signed long long Value::getAs(signed long long*) const {
  return getGV().IntVal.getSExtValue();
}
unsigned long long Value::getAs(unsigned long long*) const {
  return getGV().IntVal.getZExtValue(); 
}

} // namespace cling
