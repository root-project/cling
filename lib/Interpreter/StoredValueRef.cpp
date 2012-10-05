//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: ValuePrinter.cpp 46307 2012-10-04 06:53:23Z axel $
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/StoredValueRef.h"
#include "clang/AST/ASTContext.h"

using namespace cling;
using namespace clang;
using namespace llvm;

StoredValueRef::StoredValue::StoredValue(const ASTContext& ctx, QualType t):
m_Mem(0) {
  type = t;
  if (!(t->isIntegralOrEnumerationType()
        || t->isRealFloatingType()
        || t->hasPointerRepresentation())) {
    m_Mem = new char[getAllocSizeInBytes(ctx)];
    value = llvm::PTOGV(m_Mem);
  };
}

StoredValueRef::StoredValue::~StoredValue() {
  delete [] m_Mem;
}

int64_t StoredValueRef::StoredValue::getAllocSizeInBytes(
                                                  const ASTContext& ctx) const {
  return ctx.getTypeSizeInChars(type).getQuantity();
}




StoredValueRef StoredValueRef::allocate(const ASTContext& ctx, QualType t) {
  return new StoredValue(ctx, t);
}

StoredValueRef StoredValueRef::bitwiseCopy(const ASTContext& ctx,
                                           const Value& value) {
  StoredValue* SValue = new StoredValue(ctx, value.type);
  if (SValue->m_Mem) {
    memcpy(SValue->m_Mem, value.value.PointerVal,
           SValue->getAllocSizeInBytes(ctx));
  } else {
    SValue->value = value.value;
  }
  return SValue;
}

