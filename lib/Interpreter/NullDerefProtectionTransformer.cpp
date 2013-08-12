//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
// author:  Baozeng Ding <sploving1@gmail.com>
//------------------------------------------------------------------------------

#include "NullDerefProtectionTransformer.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InstIterator.h"

#include <cstdio>
#include "unistd.h"

extern "C" {
  bool shouldProceed(void *S, void *T) {
    using namespace clang;
    Sema *Sem = (Sema *)S;
    DiagnosticsEngine& Diag = Sem->getDiagnostics();
    cling::Transaction* Trans = (cling::Transaction*)T;
    // Here we will cheat a bit and assume that the warning came from the last
    // stmt, which will be in the 90% of the cases.
    CompoundStmt* CS = cast<CompoundStmt>(Trans->getWrapperFD()->getBody());
    // Skip the NullStmts.
    SourceLocation Loc = CS->getLocStart();
    for(CompoundStmt::const_reverse_body_iterator I = CS->body_rbegin(),
          E = CS->body_rend(); I != E; ++I)
      if (!isa<NullStmt>(*I)) {
        Loc = (*I)->getLocStart();
        break;
      }
    Diag.Report(Loc, diag::warn_null_ptr_deref);
    if (isatty(fileno(stdin))) {
      int input = getchar();
      getchar();
      if (input == 'y' || input == 'Y') 
        return false;
    }
    return true;
  }
}

using namespace clang;

namespace cling {

  NullDerefProtectionTransformer::NullDerefProtectionTransformer(Sema *S)
    : TransactionTransformer(S), FailBB(0), Builder(0), Inst(0) {}

  NullDerefProtectionTransformer::~NullDerefProtectionTransformer() 
  {}
  
  void NullDerefProtectionTransformer::Transform() {
    using namespace clang;
    FunctionDecl* FD = getTransaction()->getWrapperFD();
    if (!FD)
      return;

    // Copied from Interpreter.cpp;
    if (!m_MangleCtx)
      m_MangleCtx.reset(FD->getASTContext().createMangleContext());

    std::string mangledName;
    if (m_MangleCtx->shouldMangleDeclName(FD)) {
      llvm::raw_string_ostream RawStr(mangledName);
      switch(FD->getKind()) {
      case Decl::CXXConstructor:
        //Ctor_Complete,          // Complete object ctor
        //Ctor_Base,              // Base object ctor
        //Ctor_CompleteAllocating // Complete object allocating ctor (unused)
        m_MangleCtx->mangleCXXCtor(cast<CXXConstructorDecl>(FD), 
                                   Ctor_Complete, RawStr);
        break;

      case Decl::CXXDestructor:
        //Dtor_Deleting, // Deleting dtor
        //Dtor_Complete, // Complete object dtor
        //Dtor_Base      // Base object dtor
        m_MangleCtx->mangleCXXDtor(cast<CXXDestructorDecl>(FD),
                                   Dtor_Complete, RawStr);
        break;

      default :
        m_MangleCtx->mangleName(FD, RawStr);
        break;
      }
      RawStr.flush();
    } else {
      mangledName = FD->getNameAsString();
    }
    
    // Find the function in the module.
    llvm::Function* F = getTransaction()->getModule()->getFunction(mangledName);
    if (F)
      runOnFunction(*F);
  }

  llvm::BasicBlock*
  NullDerefProtectionTransformer::getTrapBB(llvm::BasicBlock* BB) {
    llvm::Function* Fn = Inst->getParent()->getParent();
    llvm::Module* Md = Fn->getParent();
    llvm::LLVMContext& ctx = Fn->getContext();

    FailBB = llvm::BasicBlock::Create(ctx, "FailBlock", Fn);
    Builder->SetInsertPoint(FailBB);

    std::vector<llvm::Type*> ArgTys;
    llvm::Type* VoidTy = llvm::Type::getInt8PtrTy(ctx);
    ArgTys.push_back(VoidTy);
    ArgTys.push_back(VoidTy);
    llvm::FunctionType* FTy 
      = llvm::FunctionType::get(llvm::Type::getInt1Ty(ctx), ArgTys, false);

    llvm::Function* CallBackFn
      = cast<llvm::Function>(Md->getOrInsertFunction("shouldProceed", FTy));

    void* SemaRef = (void*)m_Sema;
    // copied from JIT.cpp
    llvm::Constant* SemaRefCnt = 0;
    llvm::Type* constantIntTy = 0;
    if (sizeof(void*) == 4)
      constantIntTy = llvm::Type::getInt32Ty(ctx);
    else
      constantIntTy = llvm::Type::getInt64Ty(ctx);
      
    SemaRefCnt = llvm::ConstantInt::get(constantIntTy, (uintptr_t)SemaRef);
    llvm::Value* Arg1 = llvm::ConstantExpr::getIntToPtr(SemaRefCnt, VoidTy);

    Transaction* Trans = getTransaction();
    void* TransRef = (void*) Trans;
    llvm::Constant* TransRefCnt = llvm::ConstantInt::get(constantIntTy,
                                                       (uintptr_t)TransRef);
    llvm::Value* Arg2 = llvm::ConstantExpr::getIntToPtr(TransRefCnt, VoidTy);

    llvm::CallInst* CI = Builder->CreateCall2(CallBackFn, Arg1, Arg2);

    llvm::Value* TrueVl = llvm::ConstantInt::get(ctx, llvm::APInt(1, 1));
    llvm::Value* RetVl = CI;
    llvm::ICmpInst* Cmp 
      = new llvm::ICmpInst(*FailBB, llvm::CmpInst::ICMP_EQ, RetVl, TrueVl, "");

    llvm::BasicBlock *HandleBB = llvm::BasicBlock::Create(ctx,"HandleBlock", Fn);
    llvm::BranchInst::Create(HandleBB, BB, Cmp, FailBB);

    llvm::ReturnInst::Create(Fn->getContext(), HandleBB);
    return FailBB;
  }

  // Insert a cmp instruction before the "Inst" instruction to check whether the
  // argument "Arg" for the instruction "Inst" is null or not. 
  void NullDerefProtectionTransformer::instrumentInst(
    llvm::Instruction* Inst, llvm::Value* Arg) {
    llvm::BasicBlock* OldBB = Inst->getParent();
    
    // Insert a cmp instruction to check whether "Arg" is null or not.
    llvm::ICmpInst* Cmp 
      = new llvm::ICmpInst(Inst, llvm::CmpInst::ICMP_EQ, Arg,
                             llvm::Constant::getNullValue(Arg->getType()), "");

    // The block is splited into two blocks, "OldBB" and "NewBB", by the "Inst"
    // instruction. After that split, "Inst" is at the start of "NewBB". Then
    // insert a branch instruction at the end of "OldBB". If "Arg" is null,
    // it will jump to "FailBB" created by "getTrapBB" function. Else, it means
    // jump to the "NewBB".
    llvm::BasicBlock* NewBB = OldBB->splitBasicBlock(Inst);
    OldBB->getTerminator()->eraseFromParent();
    llvm::BranchInst::Create(getTrapBB(NewBB), NewBB, Cmp, OldBB);
  }

  bool NullDerefProtectionTransformer::runOnFunction(llvm::Function &F) {
    llvm::IRBuilder<> TheBuilder(F.getContext());
    Builder = &TheBuilder;

    std::vector<llvm::Instruction*> WorkList;
    for (llvm::inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i) {
      llvm::Instruction* I = &*i;
      if (llvm::isa<llvm::LoadInst>(I))
        WorkList.push_back(I);
    }

    for (std::vector<llvm::Instruction*>::iterator i = WorkList.begin(), 
           e = WorkList.end(); i != e; ++i) {
      Inst = *i;
      llvm::LoadInst* I = llvm::cast<llvm::LoadInst>(*i);

      // Find all the instructions that uses the instruction I.
      for (llvm::Value::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {

        // Check whether I is used as the first argument for a load instruction.
        // If it is, then instrument the load instruction.
        if (llvm::LoadInst* LI = llvm::dyn_cast<llvm::LoadInst>(*UI)) {
          llvm::Value* Arg = LI->getOperand(0);
          if (Arg == I)
            instrumentInst(LI, Arg);
        }

        // Check whether I is used as the second argument for a store
        // instruction. If it is, then instrument the store instruction.
        else if (llvm::StoreInst* SI = llvm::dyn_cast<llvm::StoreInst>(*UI)) {
          llvm::Value* Arg = SI->getOperand(1);
          if (Arg == I)
            instrumentInst(SI, Arg);
        }

        // Check whether I is used as the first argument for a GEP instruction.
        // If it is, then instrument the GEP instruction.
        else if (llvm::GetElementPtrInst* GEP = llvm::dyn_cast<
               llvm::GetElementPtrInst>(*UI)) {
          llvm::Value* Arg = GEP->getOperand(0);
          if (Arg == I)
            instrumentInst(GEP, Arg);
        }

        else {
          // Check whether I is used as the first argument for a call instruction.
          // If it is, then instrument the call instruction.
          llvm::CallSite CS(*UI);
          if (CS) {
            llvm::CallSite::arg_iterator i = CS.arg_begin(), e = CS.arg_end();
            if (i != e) {
              llvm::Value *Arg = *i;
              if (Arg == I)
               instrumentInst(CS.getInstruction(), Arg);
            }
          }
        }
      }
    }
    return true;
  }
} // end namespace cling
