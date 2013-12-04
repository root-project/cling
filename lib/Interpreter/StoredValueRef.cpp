//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: ValuePrinter.cpp 46307 2012-10-04 06:53:23Z axel $
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/ValuePrinter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"

using namespace cling;
using namespace clang;
using namespace llvm;

StoredValueRef::StoredValue::StoredValue(Interpreter& interp,
                                         QualType clangTy, 
                                         const llvm::Type* llvm_Ty)
  : Value(GenericValue(), clangTy, llvm_Ty), m_Mem(0), m_Interp(interp) {
  if (clangTy->isIntegralOrEnumerationType() ||
      clangTy->isRealFloatingType() ||
      clangTy->hasPointerRepresentation()) {
    return;
  };
  if (const MemberPointerType* MPT = clangTy->getAs<MemberPointerType>()) {
    if (MPT->isMemberDataPointer()) {
      return;
    }
  }
  m_Mem = m_Buf;
  const uint64_t size = (uint64_t) getAllocSizeInBytes();
  if (size > sizeof(m_Buf)) {
    m_Mem = new char[size];
  }
  setGV(llvm::PTOGV(m_Mem));
}

StoredValueRef::StoredValue::~StoredValue() {
  // Destruct the object, then delete the memory if needed.
  Destruct();
  if (m_Mem != m_Buf)
    delete [] m_Mem;
}

void* StoredValueRef::StoredValue::GetDtorWrapperPtr(CXXRecordDecl* CXXRD) {
   std::string funcname;
  {
    llvm::raw_string_ostream namestr(funcname);
    namestr << "__cling_StoredValue_Destruct_" << CXXRD;
  }
  void* dtorWrapperPtr = m_Interp.getAddressOfGlobal(funcname.c_str());
  if (dtorWrapperPtr)
    return dtorWrapperPtr;

  std::string code("extern \"C\" void ");
  std::string typeName = getClangType().getAsString();
  code += funcname + "(void*obj){((" + typeName + "*)obj)->~" + typeName +"();}";
  m_Interp.declare(code);
  return m_Interp.getAddressOfGlobal(funcname.c_str());
}

void StoredValueRef::StoredValue::Destruct() {
  // If applicable, call addr->~Type() to destruct the object.
  // template <typename T> void destr(T* obj = 0) { (T)obj->~T(); }
  // |-FunctionDecl destr 'void (struct XB *)'
  // |-TemplateArgument type 'struct XB'
  // |-ParmVarDecl obj 'struct XB *'
  // `-CompoundStmt
  //   `-CXXMemberCallExpr 'void'
  //     `-MemberExpr '<bound member function type>' ->~XB
  //       `-ImplicitCastExpr 'struct XB *' <LValueToRValue>
  //         `-DeclRefExpr  'struct XB *' lvalue ParmVar 'obj' 'struct XB *'

  const RecordType* RT = dyn_cast<RecordType>(getClangType());
  if (!RT)
    return;
  CXXRecordDecl* CXXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!CXXRD || CXXRD->hasTrivialDestructor())
    return;

  CXXRD = CXXRD->getCanonicalDecl();
  void* funcPtr = GetDtorWrapperPtr(CXXRD);
  if (!funcPtr)
    return;

  typedef void (*DtorWrapperFunc_t)(void* obj);
  DtorWrapperFunc_t wrapperFuncPtr = (DtorWrapperFunc_t) funcPtr;
  (*wrapperFuncPtr)(getAs<void*>());
}

long long StoredValueRef::StoredValue::getAllocSizeInBytes() const {
  const ASTContext& ctx = m_Interp.getCI()->getASTContext();
  return (long long) ctx.getTypeSizeInChars(getClangType()).getQuantity();
}


void StoredValueRef::dump() const {
  ASTContext& ctx = m_Value->m_Interp.getCI()->getASTContext();
  valuePrinterInternal::StreamStoredValueRef(llvm::errs(), this, ctx);
}

StoredValueRef StoredValueRef::allocate(Interpreter& interp, QualType t,
                                        const llvm::Type* llvmTy) {
  return new StoredValue(interp, t, llvmTy);
}

StoredValueRef StoredValueRef::bitwiseCopy(Interpreter& interp,
                                           const Value& value) {
  StoredValue* SValue 
    = new StoredValue(interp, value.getClangType(), value.getLLVMType());
  if (SValue->m_Mem) {
    const char* src = (const char*)value.getGV().PointerVal;
    // It's not a pointer. LLVM stores a char[5] (i.e. 5 x i8) as an i40,
    // so use that instead. We don't keep it as an int; instead, we "allocate"
    // it as a "proper" char[5] in the m_Mem. "Allocate" because it uses the
    // m_Buf, so no actual allocation happens.
    uint64_t IntVal = value.getGV().IntVal.getSExtValue();
    if (!src) src = (const char*)&IntVal;
    memcpy(SValue->m_Mem, src,
           SValue->getAllocSizeInBytes());
  } else {
    SValue->setGV(value.getGV());
  }
  return SValue;
}
