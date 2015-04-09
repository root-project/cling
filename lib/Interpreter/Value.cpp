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

// For address validation
#ifdef LLVM_ON_WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace {
  static bool isAddressValid(void* P) {
    if (!P || P == (void*)-1)
      return false;

#ifdef LLVM_ON_WIN32
    MEMORY_BASIC_INFORMATION MBI;
    if (!VirtualQuery(P, &MBI, sizeof(MBI)))
      return false;
    if (MBI.State != MEM_COMMIT)
      return false;
    return true;
#else
    // There is a POSIX way of finding whether an address can be accessed for
    // reading: write() will return EFAULT if not.
    int FD[2];
    if (pipe(FD))
      return false; // error in pipe()? Be conservative...
    int NBytes = write(FD[1], P, 1/*byte*/);
    close(FD[0]);
    close(FD[1]);
    if (NBytes != 1) {
      assert(errno == EFAULT && "unexpected pipe write error");
      return false;
    }
    return true;
#endif
  }

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

  static bool hasViableCandidateToCall(clang::LookupResult& R,
                                       const cling::Value& V) {
    if (R.empty())
      return false;
    using namespace clang;
    ASTContext& C = V.getASTContext();
    Sema& SemaR = R.getSema();
    OverloadCandidateSet overloads(SourceLocation(),
                                    OverloadCandidateSet::CSK_Normal);
    QualType Ty = V.getType().getNonReferenceType();
    if (!Ty->isPointerType())
      Ty = C.getPointerType(Ty);

    NamespaceDecl* ClingNSD = utils::Lookup::Namespace(&SemaR, "cling");
    RecordDecl* ClingValueDecl
      = dyn_cast<RecordDecl>(utils::Lookup::Named(&SemaR, "Value",
                                                  ClingNSD));
    assert(ClingValueDecl && "Declaration must be found!");
    QualType ClingValueTy = C.getTypeDeclType(ClingValueDecl);

    // The OverloadCandidateSet requires a QualType to be passed in through an
    // Expr* as part of Args. We know that we won't be using any node generated.
    // We need only an answer whether there is an overload taking these argument
    // types. We cannot afford to create useless Expr* on the AST for this
    // utility function which may be called thousands of times. Instead, we
    // create them on the stack and pretend they are on the heap. We get our
    // answer and forget about doing anything wrong.
    llvm::SmallVector<Expr, 4> exprsOnStack;
    SourceLocation noLoc;
    exprsOnStack.push_back(CXXNullPtrLiteralExpr(Ty, noLoc));
    exprsOnStack.push_back(CXXNullPtrLiteralExpr(Ty, noLoc));
    exprsOnStack.push_back(CXXNullPtrLiteralExpr(ClingValueTy, noLoc));
    llvm::SmallVector<Expr*, 4> exprsFakedOnHeap;
    exprsFakedOnHeap.push_back(&exprsOnStack[0]);
    exprsFakedOnHeap.push_back(&exprsOnStack[1]);
    exprsFakedOnHeap.push_back(&exprsOnStack[2]);
    llvm::ArrayRef<Expr*> Args = llvm::makeArrayRef(exprsFakedOnHeap.data(),
                                                    exprsFakedOnHeap.size());
    // Could trigger deserialization of decls.
    cling::Interpreter::PushTransactionRAII RAII(V.getInterpreter());
    SemaR.AddFunctionCandidates(R.asUnresolvedSet(), Args, overloads);

    OverloadCandidateSet::iterator Best;
    OverloadingResult OR = overloads.BestViableFunction(SemaR,
                                                        SourceLocation(), Best);
    return OR == OR_Success;
  }

  namespace valuePrinterInternal {
    void printValue_Default(llvm::raw_ostream& o, const Value& V);
    void printType_Default(llvm::raw_ostream& o, const Value& V);
  } // end namespace valuePrinterInternal

  void Value::print(llvm::raw_ostream& Out) const {
    // Try to find user defined printing functions:
    // cling::printType(const void* const p, TY* const u, const Value& V) and
    // cling::printValue(const void* const p, TY* const u, const Value& V)

    using namespace clang;
    Sema& SemaR = m_Interpreter->getSema();
    ASTContext& C = SemaR.getASTContext();
    NamespaceDecl* ClingNSD = utils::Lookup::Namespace(&SemaR, "cling");
    SourceLocation noLoc;
    LookupResult R(SemaR, &C.Idents.get("printType"), noLoc,
                   Sema::LookupOrdinaryName, Sema::ForRedeclaration);
    assert(ClingNSD && "There must be a valid namespace.");

    {
      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(m_Interpreter);
      SemaR.LookupQualifiedName(R, ClingNSD);
      // We commit here because the possibly deserialized decls from the lookup
      // will be needed by evaluate.
    }
    QualType ValueTy = this->getType().getNonReferenceType();
    bool ValidAddress = true;
    if (!ValueTy->isPointerType())
      ValueTy = C.getPointerType(ValueTy);
    else
       ValidAddress = isAddressValid(this->getPtr());
    ValueTy = utils::TypeName::GetFullyQualifiedType(ValueTy, getASTContext());
    PrintingPolicy Policy(m_Interpreter->getCI()->getLangOpts());
    std::string ValueTyStr = ValueTy.getAsString(Policy);
    std::string typeStr;
    std::string valueStr;

    if (ValidAddress && hasViableCandidateToCall(R, *this)) {
      // There is such a routine call, it:
      std::stringstream printTypeSS;
      printTypeSS << "cling::printType(";
      printTypeSS << '(' << ValueTyStr << ')' << this->getPtr() << ',';
      printTypeSS << '(' << ValueTyStr << ')' << this->getPtr() << ',';
      printTypeSS <<"(*(cling::Value*)" << this << "));";
      Value printTypeV;
      m_Interpreter->evaluate(printTypeSS.str(), printTypeV);
      assert(printTypeV.isValid() && "Must return valid value.");
      typeStr = *(std::string*)printTypeV.getPtr();
      // CXXScopeSpec CSS;
      // Expr* UnresolvedLookup
      //   = m_Sema->BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).take();
      // // Build Arg1: const void* const p
      // QualType ConstVoidPtrTy = C.VoidPtrTy.withConst();
      // Expr* Arg1
      //   = utils::Synthesize::CStyleCastPtrExpr(SemaR, ConstVoidPtrTy,
      //                                          (uint64_t)this->getPtr());

      // // Build Arg2: TY* const u
      // Expr* Arg2
      //   = utils::Synthesize::CStyleCastPtrExpr(SemaR, ValueTy,
      //                                          (uint64_t)this->getPtr());

      // // Build Arg3: const Value&
      // RecordDecl* ClingValueDecl
      //   = dyn_cast<RecordDecl>(utils::Lookup::Named(SemaR, "Value",ClingNSD));
      // assert(ClingValueDecl && "Declaration must be found!");
      // QualType ClingValueTy = m_Context->getTypeDeclType(ClingValueDecl);
      // Expr* Arg3
      //   = utils::Synthesize::CStyleCastPtrExpr(m_Sema, ClingValueTy,
      //                                          (uint64_t)this);
      // llvm::SmallVector<Expr*, 4> CallArgs;
      // CallArgs.push_back(Arg1);
      // CallArgs.push_back(Arg2);
      // CallArgs.push_back(Arg3);
      // Expr* Call = m_Sema->ActOnCallExpr(/*Scope*/0, UnresolvedLookup, noLoc,
      //                                    CallArgs, noLoc).take();
    }
    else {
      llvm::raw_string_ostream o(typeStr);
      cling::valuePrinterInternal::printType_Default(o, *this);
    }
    R.clear();
    R.setLookupName(&C.Idents.get("printValue"));
    {
      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(m_Interpreter);
      SemaR.LookupQualifiedName(R, ClingNSD);
      // We commit here because the possibly deserialized decls from the lookup
      // will be needed by evaluate.
    }

    if (ValidAddress && hasViableCandidateToCall(R, *this)) {
      // There is such a routine call it:
      std::stringstream printValueSS;
      printValueSS << "cling::printValue(";
      printValueSS << '(' << ValueTyStr << ')' << this->getPtr() << ',';
      printValueSS << '(' << ValueTyStr << ')' << this->getPtr() << ',';
      printValueSS <<"(*(cling::Value*)" << this << "));";
      Value printValueV;
      m_Interpreter->evaluate(printValueSS.str(), printValueV);
      assert(printValueV.isValid() && "Must return valid value.");
      valueStr = *(std::string*)printValueV.getPtr();
    }
    else {
      llvm::raw_string_ostream o(valueStr);
      cling::valuePrinterInternal::printValue_Default(o, *this);
    }

    // print the type and the value:
    Out << typeStr + valueStr << "\n";
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
