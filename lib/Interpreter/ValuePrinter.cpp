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

using namespace cling;

// Implements the CValuePrinter interface.
extern "C" void cling_PrintValue(void* /*clang::Expr**/ E,
                      void* /*clang::ASTContext**/ C,
                      const void* value) {
  clang::Expr* Exp = (clang::Expr*)E;
  clang::ASTContext* Context = (clang::ASTContext*)C;
  ValuePrinterInfo VPI(Exp->getType(), Context);
  printValuePublic(llvm::outs(), value, value, VPI);

  flushOStream(llvm::outs());
}


static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const ValuePrinterInfo& VPI,
                        const char* Sep = "\n");

static void StreamChar(llvm::raw_ostream& o, const char v,
                       const char* Sep = "\n") {
  o << '"' << v << "\"" << Sep;
}

static void StreamCharPtr(llvm::raw_ostream& o, const char* const v,
                          const char* Sep = "\n") {
  if (!v) {
    o << "<<<NULL>>>" << Sep;
    return;
  }
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

static void StreamArr(llvm::raw_ostream& o, const void* p,
                      const ValuePrinterInfo& VPI,
                      const char* Sep = "\n") {
  const clang::QualType& Ty = VPI.getType();
  clang::ASTContext& C = *VPI.getASTContext();
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
    ValuePrinterInfo ElVPI(ElementTy, &C);
    for (size_t i = 0; i < Size; ++i) {
      StreamValue(o, ((const char*)p) + i * ElBytes, ElVPI, "");
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

static void StreamClingValue(llvm::raw_ostream& o, const Value* value,
                             clang::ASTContext& C, const char* Sep = "\n") {
  if (!value || !value->isValid()) {
    o << "<<<invalid>>> @" << value << Sep;
  } else {
    o << "boxes [";
    o << "("
      << value->type.getAsString(C.getPrintingPolicy())
      << ") ";
    clang::QualType valType = value->type.getDesugaredType(C);
    if (valType->isFloatingType())
      o << value->value.DoubleVal;
    else if (valType->isIntegerType())
      o << value->value.IntVal.getSExtValue();
    else if (valType->isBooleanType())
      o << value->value.IntVal.getBoolValue();
    else
      StreamValue(o, value->value.PointerVal,
                  ValuePrinterInfo(valType, &C), "");
    o << "]" << Sep;
  }
}

void cling::StreamStoredValueRef(llvm::raw_ostream& o, const StoredValueRef* VR,
                                 clang::ASTContext& C,
                                 const char* Sep /*= "\n"*/) {

  if (VR->isValid()) {
    StreamClingValue(o, &VR->get(), C, Sep);
  } else {
    o << "<<<invalid>>> @" << VR << Sep;
  }
}

static void StreamObj(llvm::raw_ostream& o, const void* v,
                      const ValuePrinterInfo& VPI,
                      const char* Sep = "\n") {
  const clang::Type* Ty = VPI.getType().getTypePtr();
  if (clang::CXXRecordDecl* CXXRD = Ty->getAsCXXRecordDecl()) {
    std::string QualName = CXXRD->getQualifiedNameAsString();
    if (QualName == "cling::StoredValueRef"){
      StreamStoredValueRef(o, (const StoredValueRef*)v,
                           *VPI.getASTContext(), Sep);
      return;
    } else if (QualName == "cling::Value") {
      StreamClingValue(o, (const Value*)v, *VPI.getASTContext(), Sep);
      return;
    }
  } // if CXXRecordDecl

  // TODO: Print the object members.
  o << "@" << v << Sep;
}

static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const ValuePrinterInfo& VPI,
                        const char* Sep /*= "\n"*/) {
  clang::ASTContext& C = *VPI.getASTContext();
  clang::QualType Ty = VPI.getType().getDesugaredType(C);
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
      StreamObj(o, p, ValuePrinterInfo(Ty, &C), Sep);
    }
  }
  else if (Ty.getAsString().compare("class std::basic_string<char>") == 0) {
    StreamObj(o, p, ValuePrinterInfo(Ty, &C), Sep);
    if (!Sep[0]) o << " "; // force a space
    o <<"c_str: ";
    StreamCharPtr(o, ((const char*) (*(const std::string*)p).c_str()), Sep);
  }
  else if (Ty->isEnumeralType()) {
    StreamObj(o, p, ValuePrinterInfo(Ty, &C), Sep);
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
    StreamArr(o, p, ValuePrinterInfo(Ty, &C), Sep);
  else
    StreamObj(o, p, ValuePrinterInfo(Ty, &C), Sep);
}

namespace cling {
  void printValuePublicDefault(llvm::raw_ostream& o, const void* const p,
                               const ValuePrinterInfo& VPI) {
    o << "(";
    o << VPI.getType().getAsString();
    o << ") ";
    StreamValue(o, p, VPI);
  }

  void flushOStream(llvm::raw_ostream& o) {
    o.flush();
  }

} // end namespace cling
