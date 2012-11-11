//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/ValuePrinter.h"

#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/ValuePrinterInfo.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Value.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

#include "llvm/Support/raw_ostream.h"

#include <string>

// Implements the CValuePrinter interface.
extern "C" void cling_PrintValue(void* /*clang::Expr**/ E,
                      void* /*clang::ASTContext**/ C,
                      const void* value) {
  clang::Expr* Exp = (clang::Expr*)E;
  clang::ASTContext* Context = (clang::ASTContext*)C;
  cling::ValuePrinterInfo VPI(Exp, Context);
  cling::printValuePublic(llvm::outs(), value, value, VPI);

  cling::flushOStream(llvm::outs());
}


static void StreamChar(llvm::raw_ostream& o, const char v,
                       const char* Sep = "\n") {
  o << '"' << v << "\"" << Sep;
}

static void StreamCharPtr(llvm::raw_ostream& o, const char* const v,
                          const char* Sep = "\n") {
  o << '"';
  const char* p = v;
  for (;*p && p - v < 128; ++p) {
    o << *p;
  }
  if (*p) o << "\"..." << Sep;
  else o << "\"" << Sep;
}

static void StreamRef(llvm::raw_ostream& o, const void* v,
                      const char* Sep = "\n") {
  o <<"&" << v << Sep;
}

static void StreamPtr(llvm::raw_ostream& o, const void* v,
                      const char* Sep = "\n") {
  o << v << Sep;
}

static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const cling::ValuePrinterInfo& VPI,
                        clang::QualType Ty,
                        const char* Sep = "\n");

static void StreamArr(llvm::raw_ostream& o, const void* p,
                      const cling::ValuePrinterInfo& VPI,
                      clang::QualType Ty,
                      const char* Sep = "\n") {
  const clang::ASTContext& C = *VPI.getASTContext();
  const clang::ArrayType* ArrTy = Ty->getAsArrayTypeUnsafe();
  clang::QualType ElementTy = ArrTy->getElementType();
  if (ElementTy->isCharType())
    StreamCharPtr(o, (const char*)p);
  else if (Ty->isConstantArrayType()) {
    // Stream a constant array by streaming up to 5 elements.
    const clang::ConstantArrayType* CArrTy
      = C.getAsConstantArrayType(Ty);
    const llvm::APInt& APSize = CArrTy->getSize();
    size_t ElBytes = C.getTypeSize(ElementTy) / C.getCharWidth();
    size_t Size = (size_t)APSize.getZExtValue();
    o << "{ ";
    for (size_t i = 0; i < Size; ++i) {
      StreamValue(o, ((const char*)p) + i * ElBytes, VPI, ElementTy, " ");
      if (i + 1 < Size) {
        if (i == 4) {
          o << "...";
          break;
        }
        else o << ", ";
      }
    }
    o << " }" << Sep;
  } else
    StreamPtr(o, p, Sep);
}

static void StreamObj(llvm::raw_ostream& o, const void* v,
                      const cling::ValuePrinterInfo& VPI,
                      clang::QualType QTy,
                      const char* Sep = "\n") {
  const clang::Type* Ty = QTy.getTypePtr();
  if (clang::CXXRecordDecl* CXXRD = Ty->getAsCXXRecordDecl()) {
    const cling::Value* value = 0;
    if (CXXRD->getQualifiedNameAsString().compare("cling::StoredValueRef") == 0){
      const cling::StoredValueRef* VR = (const cling::StoredValueRef*)v;
      if (VR->isValid()) {
        value = &VR->get();
      } else {
        o << "<<<invalid>>> ";
      }
    } else if (CXXRD->getQualifiedNameAsString().compare("cling::Value") == 0) {
      value = (const cling::Value*)v;
    }
    if (value) {
      if (value->isValid()) {
        o << "boxes [";
        const clang::ASTContext& C = *VPI.getASTContext();
        o <<
          "(" <<
          value->type.getAsString(C.getPrintingPolicy()) <<
          ") ";
        clang::QualType valType = value->type.getDesugaredType(C);
        if (valType->isFloatingType())
          o << value->value.DoubleVal;
        else if (valType->isIntegerType())
          o << value->value.IntVal.getSExtValue();
        else if (valType->isBooleanType())
          o << value->value.IntVal.getBoolValue();
        else
          StreamValue(o, value->value.PointerVal, VPI, valType, "");
        o << "]" << Sep;

        return;
      } else
        o << "<<<invalid>>> ";
    }
  } // if CXXRecordDecl

  // TODO: Print the object members.
  o << "@" << v << Sep;
}

static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const cling::ValuePrinterInfo& VPI,
                        clang::QualType Ty,
                        const char* Sep /*= "\n"*/) {
  const clang::ASTContext& C = *VPI.getASTContext();
  Ty = Ty.getDesugaredType(C);
  if (const clang::BuiltinType *BT
           = llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
    switch (BT->getKind()) {
    case clang::BuiltinType::Bool:
      if (*(const bool*)p) o << "true" << Sep;
      else o << "false" << Sep; break;
    case clang::BuiltinType::Char_U:
    case clang::BuiltinType::UChar:
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::SChar:  StreamChar(o, *(const char*)p); break;
    case clang::BuiltinType::Short:  o << *(const short*)p << Sep; break;
    case clang::BuiltinType::UShort:
      o << *(const unsigned short*)p << Sep;
      break;
    case clang::BuiltinType::Int:    o << *(const int*)p << Sep; break;
    case clang::BuiltinType::UInt:
      o << *(const unsigned int*)p << Sep;
      break;
    case clang::BuiltinType::Long:   o << *(const long*)p << Sep; break;
    case clang::BuiltinType::ULong:
      o << *(const unsigned long*)p << Sep;
      break;
    case clang::BuiltinType::LongLong:
      o << *(const long long*)p << Sep;
      break;
    case clang::BuiltinType::ULongLong:
      o << *(const unsigned long long*)p << Sep;
      break;
    case clang::BuiltinType::Float:  o << *(const float*)p << Sep; break;
    case clang::BuiltinType::Double: o << *(const double*)p << Sep; break;
    default:
      StreamObj(o, p, VPI, Ty, Sep);
    }
  }
  else if (Ty.getAsString().compare("class std::basic_string<char>") == 0) {
    StreamObj(o, p, VPI, Ty, Sep);
    o <<"c_str: ";
    StreamCharPtr(o, ((const char*) (*(const std::string*)p).c_str()), Sep);
  }
  else if (Ty->isEnumeralType()) {
    StreamObj(o, p, VPI, Ty, Sep);
    clang::EnumDecl* ED = Ty->getAs<clang::EnumType>()->getDecl();
    uint64_t value = *(const uint64_t*)p;
    bool IsFirst = true;
    llvm::APSInt ValAsAPSInt = C.MakeIntValue(value, Ty);
    for (clang::EnumDecl::enumerator_iterator I = ED->enumerator_begin(),
           E = ED->enumerator_end(); I != E; ++I) {
      if (I->getInitVal() == ValAsAPSInt) {
        if (!IsFirst) {
          o << " ? ";
        }
        o << "(" << I->getQualifiedNameAsString() << ")";
        IsFirst = false;
      }
    }
    o << " : (int) " << ValAsAPSInt.toString(/*Radix = */10) << Sep;
  }
  else if (Ty->isReferenceType())
    StreamRef(o, p, Sep);
  else if (Ty->isPointerType()) {
    clang::QualType PointeeTy = Ty->getPointeeType();
    if (PointeeTy->isCharType())
      StreamCharPtr(o, (const char*)p, Sep);
    else
      StreamPtr(o, p, Sep);
  }
  else if (Ty->isArrayType())
    StreamArr(o, p, VPI, Ty, Sep);
  else
    StreamObj(o, p, VPI, Ty, Sep);
}

namespace cling {
  void printValuePublicDefault(llvm::raw_ostream& o, const void* const p,
                               const ValuePrinterInfo& VPI) {
    const clang::Expr* E = VPI.getExpr();
    o << "(";
    o << E->getType().getAsString();
    o << ") ";
    StreamValue(o, p, VPI, VPI.getExpr()->getType());
  }

  void flushOStream(llvm::raw_ostream& o) {
    o.flush();
  }

} // end namespace cling
