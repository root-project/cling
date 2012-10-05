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
Mem(0) {
  type = t;
  if (!(t->isIntegralOrEnumerationType()
        || t->isRealFloatingType()
        || t->hasPointerRepresentation())) {
    Mem = new char[getAllocSizeInBytes(ctx)];
    value = llvm::PTOGV(Mem);
  };
}

StoredValueRef::StoredValue::~StoredValue() {
  delete [] Mem;
}

int64_t StoredValueRef::StoredValue::getAllocSizeInBytes(
                                                  const ASTContext& ctx) const {
  return ctx.getTypeSizeInChars(type).getQuantity();
}

void StoredValueRef::StoredValue::adopt(char* addr) {
  delete [] Mem;
  Mem = addr;
  value.PointerVal = Mem;
}




StoredValueRef StoredValueRef::allocate(const ASTContext& ctx, QualType t) {
  return new StoredValue(ctx, t);
}

StoredValueRef StoredValueRef::bitwiseCopy(const ASTContext& ctx,
                                           const Value& value) {
  StoredValue* SValue = new StoredValue(ctx, value.type);
  if (SValue->Mem) {
    memcpy(SValue->Mem, value.value.PointerVal,
           SValue->getAllocSizeInBytes(ctx));
  } else {
    SValue->value = value.value;
  }
  return SValue;
}

