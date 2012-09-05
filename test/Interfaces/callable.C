// RUN: cat %s | %cling | FileCheck %s
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/Value.h"
#include "cling/Interpreter/Callable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTContext.h"

.rawInput
extern "C" int printf(const char*,...);
void Bogus(int i) { printf("Bogus got a %d!\n", i);}
class Smart {
public:
  Smart(): I(42) {}
  int Inline() { printf("Inline!\n"); return I + 1; }
  int TheAnswer() const;
  int I;
};
int Smart::TheAnswer() const{ printf("TheAnswer!\n"); return I; }
.rawInput

const clang::Decl* TU = gCling->getCI()->getASTContext().getTranslationUnitDecl();
const cling::LookupHelper& lookup = gCling->getLookupHelper();

const clang::FunctionDecl* FBogus = lookup.findFunctionProto(TU, "Bogus", "int");
assert(FBogus && "Bogus() not found");
cling::Callable CBogus(*FBogus, *gCling);
std::vector<llvm::GenericValue> ArgVs;
ArgVs.push_back(llvm::GenericValue());
ArgVs[0].IntVal = llvm::APInt(sizeof(int)*8, 12);
CBogus.Invoke(ArgVs); // CHECK: Bogus got a 12!
cling::Value V;
++ArgVs[0].IntVal;
CBogus.Invoke(ArgVs, &V); // CHECK: Bogus got a 13!
V // CHECK: (cling::Value) boxes [(void) ]

const clang::Decl* DSmart = lookup.findScope("Smart");
assert(DSmart && "Smart not found");
const clang::FunctionDecl* FAnswer = lookup.findFunctionProto(DSmart, "TheAnswer", "");
assert(FAnswer && "Smart::TheAnswer() not found");
cling::Callable CAnswer(*FAnswer,*gCling);
Smart s;
ArgVs[0] = llvm::GenericValue(&s);
if (!CAnswer.Invoke(ArgVs, &V)) {printf("CAnswer failed!\n");} // CHECK: TheAnswer!
V // CHECK: (cling::Value) boxes [(int) 42]

const clang::FunctionDecl* FInline = lookup.findFunctionProto(DSmart, "Inline", "");
assert(FInline && "Smart::Inline() not found");
cling::Callable CInline(*FInline,*gCling);
if (!CInline.Invoke(ArgVs, &V)) {printf("CInline failed!\n");} // CHECK: CInline failed!
V // CHECK: (cling::Value) boxes [(int) 42]
