//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: ValuePrinter.cpp 46307 2012-10-04 06:53:23Z axel $
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/ValuePrinter.h"
#include "clang/AST/ASTContext.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/raw_ostream.h"

using namespace cling;
using namespace clang;
using namespace llvm;

StoredValueRef::StoredValue::StoredValue(const ASTContext& ctx, 
                                         QualType clangTy, 
                                         const llvm::Type* llvm_Ty)
   :Value(GenericValue(), clangTy, llvm_Ty), m_Mem(0) {
  if (!(clangTy->isIntegralOrEnumerationType()
        || clangTy->isRealFloatingType()
        || clangTy->hasPointerRepresentation())) {
    const uint64_t size = (uint64_t)getAllocSizeInBytes(ctx);
    if (size > sizeof(m_Buf))
      m_Mem = new char[size];
    else m_Mem = m_Buf;
    setGV(llvm::PTOGV(m_Mem));
  };
}

StoredValueRef::StoredValue::~StoredValue() {
  if (m_Mem != m_Buf)
    delete [] m_Mem;
}

long long StoredValueRef::StoredValue::getAllocSizeInBytes(
                                                  const ASTContext& ctx) const {
   return (long long) ctx.getTypeSizeInChars(getClangType()).getQuantity();
}


void StoredValueRef::dump(ASTContext& ctx) const {
  StreamStoredValueRef(llvm::errs(), this, ctx);
}

StoredValueRef StoredValueRef::allocate(const ASTContext& ctx, QualType t, 
                                        const llvm::Type* llvmTy) {
  return new StoredValue(ctx, t, llvmTy);
}

StoredValueRef StoredValueRef::bitwiseCopy(const ASTContext& ctx,
                                           const Value& value) {
  StoredValue* SValue 
    = new StoredValue(ctx, value.getClangType(), value.getLLVMType());
  if (SValue->m_Mem) {
    const char* src = (const char*)value.getGV().PointerVal;
    // It's not a pointer. LLVM stores a char[5] (i.e. 5 x i8) as an i40,
    // so use that instead. We don't keep it as an int; instead, we "allocate"
    // it as a "proper" char[5] in the m_Mem. "Allocate" because it uses the
    // m_Buf, so no actual allocation happens.
    uint64_t IntVal = value.getGV().IntVal.getSExtValue();
    if (!src) src = (const char*)&IntVal;
    memcpy(SValue->m_Mem, src,
           SValue->getAllocSizeInBytes(ctx));
  } else {
    SValue->setGV(value.getGV());
  }
  return SValue;
}
