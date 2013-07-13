//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
// author:  Baozeng Ding <sploving1@gmail.com>
//------------------------------------------------------------------------------

#ifndef CLING_NULL_DEREFERENCE_PROTECTION_TRANSFORMER
#define CLING_NULL_DEREFERENCE_PROTECTION_TRANSFORMER

#include "TransactionTransformer.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {
  class MangleContext;
}

namespace llvm {
  class BasicBlock;
  class Function;
  class LoadInst;
  class StoreInst;
}

namespace cling {

  struct NullDerefProtectionTransformer : public TransactionTransformer {
  
   private:
    llvm::BasicBlock* FailBB;
    llvm::IRBuilder<>* Builder;
    llvm::Instruction* Inst;
    llvm::OwningPtr<clang::MangleContext> m_MangleCtx;
    
    llvm::BasicBlock* getTrapBB();
    void instrumentLoadInst(llvm::LoadInst* LI);
    void instrumentStoreInst(llvm::StoreInst* SI);
    bool runOnFunction(llvm::Function& F);

  public:
    NullDerefProtectionTransformer();
    virtual ~NullDerefProtectionTransformer();

    virtual void Transform();

 
  };
} // end namespace cling

#endif // CLING_NULL_DEREFERENCE_PROTECTION_TRANSFORMER
