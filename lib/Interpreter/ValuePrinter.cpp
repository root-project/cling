//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/ValuePrinter.h"

#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/ValuePrinterInfo.h"
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


static void StreamChar(llvm::raw_ostream& o, const char v) {
  o << '"' << v << "\"\n";
}

static void StreamCharPtr(llvm::raw_ostream& o, const char* const v) {
  o << '"';
  const char* p = v;
  for (;*p && p - v < 128; ++p) {
    o << *p;
  }
  if (*p) o << "\"...\n";
  else o << "\"\n";
}

static void StreamRef(llvm::raw_ostream& o, const void* v) {
  o <<"&" << v << "\n";
}

static void StreamPtr(llvm::raw_ostream& o, const void* v) {
  o << v << "\n";
}

static void StreamObj(llvm::raw_ostream& o, const void* v,
                      const cling::ValuePrinterInfo& VPI) {
  const clang::Type* Ty = VPI.getExpr()->getType().getTypePtr();
  if (clang::CXXRecordDecl* CXXRD = Ty->getAsCXXRecordDecl())
    if (CXXRD->getQualifiedNameAsString().compare("cling::Value") == 0) {
      cling::Value* V = (cling::Value*)v;
      if (V->isValid()) {
        o << "boxes [";
        const clang::ASTContext& C = *VPI.getASTContext();
        o <<
          "(" <<
          V->type.getAsString(C.getPrintingPolicy()) <<
          ") ";
        clang::QualType valType = V->type.getDesugaredType(C);
        if (valType->isPointerType())
          o << V->value.PointerVal;
        else if (valType->isFloatingType())
          o << V->value.DoubleVal;
        else if (valType->isIntegerType())
          o << V->value.IntVal.getSExtValue();
        else if (valType->isBooleanType())
          o << V->value.IntVal.getBoolValue();
        o << "]\n";

        return;
      } else
        o << "<<<invalid>>> ";
    }

  // TODO: Print the object members.
  o << "@" << v << "\n";
}

static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const cling::ValuePrinterInfo& VPI) {
  clang::QualType Ty = VPI.getExpr()->getType();
  const clang::ASTContext& C = *VPI.getASTContext();
  Ty = Ty.getDesugaredType(C);
  if (const clang::BuiltinType *BT
           = llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
    switch (BT->getKind()) {
    case clang::BuiltinType::Bool:
      if (*(bool*)p) o << "true\n";
      else o << "false\n"; break;
    case clang::BuiltinType::Char_U:
    case clang::BuiltinType::UChar:
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::SChar:  StreamChar(o, *(char*)p); break;
    case clang::BuiltinType::Short:  o << *(short*)p << "\n"; break;
    case clang::BuiltinType::UShort: o << *(unsigned short*)p << "\n"; break;
    case clang::BuiltinType::Int:    o << *(int*)p << "\n"; break;
    case clang::BuiltinType::UInt:   o << *(unsigned int*)p << "\n"; break;
    case clang::BuiltinType::Long:   o << *(long*)p << "\n"; break;
    case clang::BuiltinType::ULong:  o << *(unsigned long*)p << "\n"; break;
    case clang::BuiltinType::LongLong:  o << *(long long*)p << "\n"; break;
    case clang::BuiltinType::ULongLong: o << *(unsigned long long*)p << "\n";
      break;
    case clang::BuiltinType::Float:  o << *(float*)p << "\n"; break;
    case clang::BuiltinType::Double: o << *(double*)p << "\n"; break;
    default:
      StreamObj(o, p, VPI);
    }
  }
  else if (Ty.getAsString().compare("class std::basic_string<char>") == 0) {
    StreamObj(o, p, VPI);
    o <<"c_str: ";
    StreamCharPtr(o, ((const char*) (*(const std::string*)p).c_str()));
  }
  else if (Ty->isEnumeralType()) {
    StreamObj(o, p, VPI);
    clang::EnumDecl* ED = Ty->getAs<clang::EnumType>()->getDecl();
    uint64_t value = *(uint64_t*)p;
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
    o << " : (int) " << ValAsAPSInt.toString(/*Radix = */10) << "\n";
  }
  else if (Ty->isReferenceType())
    StreamRef(o, p);
  else if (Ty->isPointerType()) {
    clang::QualType PointeeTy = Ty->getPointeeType();
    if (PointeeTy->isCharType())
      StreamCharPtr(o, (const char*)p);
    else
      StreamPtr(o, p);
  }
  else
    StreamObj(o, p, VPI);
}

namespace cling {
  void printValuePublicDefault(llvm::raw_ostream& o, const void* const p,
                               const ValuePrinterInfo& VPI) {
    const clang::Expr* E = VPI.getExpr();
    o << "(";
    o << E->getType().getAsString();
    if (E->isRValue()) // show the user that the var cannot be changed
      o << " const";
    o << ") ";
    StreamValue(o, p, VPI);
  }

  void flushOStream(llvm::raw_ostream& o) {
    o.flush();
  }

} // end namespace cling
