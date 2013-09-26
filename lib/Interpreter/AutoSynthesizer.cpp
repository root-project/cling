//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "AutoSynthesizer.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  class AutoFixer : public RecursiveASTVisitor<AutoFixer> {
  private:
    Sema* m_Sema;

  private:
    inline bool isAutoCandidate(const BinaryOperator* BinOp) const {
      assert(BinOp && "Cannot be null.");
      Expr* LHS = BinOp->getLHS();
      if (const DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(LHS)) {
        const Decl* D = DRE->getDecl();
        if (const AnnotateAttr* A = D->getAttr<AnnotateAttr>())
          if (A->getAnnotation().equals("__Auto"))
            return true;
      }
      return false;
    }
  public:
    AutoFixer(Sema* S) : m_Sema(S) {}

    void Fix(CompoundStmt* CS) {
      TraverseStmt(CS);
    }

    bool VisitCompoundStmt(CompoundStmt* CS) {
      for(CompoundStmt::body_iterator I = CS->body_begin(), E = CS->body_end();
          I != E; ++I) {
        if (!isa<BinaryOperator>(*I))
          continue;

        const BinaryOperator* BinOp = cast<BinaryOperator>(*I);
        if (isAutoCandidate(BinOp)) {
          ASTContext& C = m_Sema->getASTContext();
          VarDecl* VD 
            = cast<VarDecl>(cast<DeclRefExpr>(BinOp->getLHS())->getDecl());
          QualType ResTy;
          struct NonDependentSetter: public clang::Type {
            static void set(clang::QualType QT) {
              clang::Type* Ty = const_cast<clang::Type*>(QT.getTypePtr());
              static_cast<NonDependentSetter*>(Ty)->setDependent(false);
            }
          };
          NonDependentSetter::set(VD->getType());
          TypeSourceInfo* TrivialTSI
            = C.getTrivialTypeSourceInfo(VD->getType());
          Expr* RHS = BinOp->getRHS();
          m_Sema->DeduceAutoType(TrivialTSI, RHS, ResTy);
          VD->setTypeSourceInfo(C.getTrivialTypeSourceInfo(ResTy));
          VD->setType(ResTy);
          VD->setInit(RHS);
          Sema::DeclGroupPtrTy VDPtrTy = m_Sema->ConvertDeclToDeclGroup(VD);
          // Transform the AST into a "sane" state. Replace the binary operator
          // with decl stmt, because the binop semantically is a decl with init.
          StmtResult DS = m_Sema->ActOnDeclStmt(VDPtrTy, BinOp->getLocStart(), 
                                                BinOp->getLocEnd());
          assert(!DS.isInvalid() && "Invalid DeclStmt.");
          *I = DS.take();
        }
      }
      return true; // returning false will abort the in-depth traversal.
    }
  };
} // end namespace cling

namespace cling { 
  AutoSynthesizer::AutoSynthesizer(clang::Sema* S)
    : TransactionTransformer(S) {
  }

  // pin the vtable here.
  AutoSynthesizer::~AutoSynthesizer() 
  { }

  void AutoSynthesizer::Transform() {
    // if (!getTransaction()->getCompilationOpts().ResultEvaluation)
    //   return;
    AutoFixer autoFixer(m_Sema);
    // size can change in the loop!
    for (size_t Idx = 0; Idx < getTransaction()->size(); ++Idx) {
      Transaction::DelayCallInfo I = (*getTransaction())[Idx];
      for (DeclGroupRef::const_iterator J = I.m_DGR.begin(), 
             JE = I.m_DGR.end(); J != JE; ++J)
        if ((*J)->hasBody())
          autoFixer.Fix(cast<CompoundStmt>((*J)->getBody()));
    }
  }
} // end namespace cling
