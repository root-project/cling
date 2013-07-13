//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasielv@cern.ch>
// author:  Baozeng Ding <sploving1@gmail.com>
//------------------------------------------------------------------------------

#include "NullDerefProtectionTransformer.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InstIterator.h"

namespace cling {

  NullDerefProtectionTransformer::NullDerefProtectionTransformer()
    : FailBB(0), TransactionTransformer(/*Sema=*/0) {}

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
                                   Dtor_Deleting, RawStr);
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
    llvm::Function* F
      = getTransaction()->getModule()->getFunction(mangledName);
    if (F)
      runOnFunction(*F);
  }

  llvm::BasicBlock* NullDerefProtectionTransformer::getTrapBB() {

    if (FailBB) return FailBB;
    llvm::Function *Fn = Inst->getParent()->getParent();

    llvm::LLVMContext& ctx = Fn->getContext();

    FailBB = llvm::BasicBlock::Create(ctx, "FailBlock", Fn);
    llvm::ReturnInst::Create(Fn->getContext(), FailBB);
    return FailBB;
  }

  void NullDerefProtectionTransformer::instrumentLoadInst(llvm::LoadInst *LI) {
    LI->dump();
    llvm::Value * Addr = LI->getOperand(0);
    Addr->dump();
    LI->getParent()->getParent()->dump();
    llvm::PointerType* PTy = llvm::cast<llvm::PointerType>(Addr->getType());
    llvm::Type * ElTy = PTy -> getElementType();
    if (!ElTy->isPointerTy()) {
      llvm::BasicBlock *OldBB = LI->getParent();
      llvm::ICmpInst *Cmp 
        = new llvm::ICmpInst(LI, llvm::CmpInst::ICMP_EQ, Addr,
                             llvm::Constant::getNullValue(Addr->getType()), "");

      llvm::Instruction *Inst = Builder->GetInsertPoint();
      llvm::BasicBlock *NewBB = OldBB->splitBasicBlock(Inst);
       
      OldBB->getTerminator()->eraseFromParent();
      llvm::BranchInst::Create(getTrapBB(), NewBB, Cmp, OldBB);
    }
  }

  void NullDerefProtectionTransformer::instrumentStoreInst(llvm::StoreInst *SI){
    llvm::Value * Addr = SI->getOperand(1);
    llvm::PointerType* PTy = llvm::cast<llvm::PointerType>(Addr->getType());
    llvm::Type * ElTy = PTy -> getElementType();
    if (!ElTy->isPointerTy()) {
      llvm::BasicBlock *OldBB = SI->getParent();
      llvm::ICmpInst *Cmp 
        = new llvm::ICmpInst(SI, llvm::CmpInst::ICMP_EQ, Addr,
                             llvm::Constant::getNullValue(Addr->getType()), "");

      llvm::Instruction *Inst = Builder->GetInsertPoint();
      llvm::BasicBlock *NewBB = OldBB->splitBasicBlock(Inst);
       
      OldBB->getTerminator()->eraseFromParent();
      llvm::BranchInst::Create(getTrapBB(), NewBB, Cmp, OldBB);
    }
  }

  bool NullDerefProtectionTransformer::runOnFunction(llvm::Function &F) {
    llvm::IRBuilder<> TheBuilder(F.getContext());
    Builder = &TheBuilder;
      
    std::vector<llvm::Instruction*> WorkList;
    for (llvm::inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i) {
      llvm::Instruction *I = &*i;
      if (llvm::isa<llvm::LoadInst>(I) || llvm::isa<llvm::StoreInst>(I))
        WorkList.push_back(I);
      }

    for (std::vector<llvm::Instruction*>::iterator i = WorkList.begin(), 
           e = WorkList.end(); i != e; ++i) {
      Inst = *i;

      Builder->SetInsertPoint(Inst);
      if (llvm::LoadInst *LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        instrumentLoadInst(LI);
        LI->dump();
      } else if (llvm::StoreInst *SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
        instrumentStoreInst(SI);
        SI->dump();
      } else {
        llvm_unreachable("unknown Instruction type");
      }
    }
    return true;
  }
} // end namespace cling
