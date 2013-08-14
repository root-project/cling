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
  class Sema;
  class StoreInst;
}

namespace cling {

  struct NullDerefProtectionTransformer : public TransactionTransformer {
  
   private:
    llvm::BasicBlock* FailBB;
    llvm::IRBuilder<>* Builder;
    llvm::Instruction* Inst;
    llvm::OwningPtr<clang::MangleContext> m_MangleCtx;

    llvm::BasicBlock* getTrapBB(llvm::BasicBlock* BB);
    void instrumentInst(llvm::Instruction* Inst, llvm::Value* Arg);
    bool runOnFunction(llvm::Function& F);
    void instrumentCallInst(llvm::Instruction* TheCall,
                            const std::bitset<32>& ArgIndexs);
    void handleNonNullArgCall(llvm::Module& M,
                              const llvm::StringRef& FName,
                              const std::bitset<32>& ArgIndexs);

  public:
    NullDerefProtectionTransformer(clang::Sema* S);
    virtual ~NullDerefProtectionTransformer();

    virtual void Transform();
  };
} // end namespace cling

#endif // CLING_NULL_DEREFERENCE_PROTECTION_TRANSFORMER
