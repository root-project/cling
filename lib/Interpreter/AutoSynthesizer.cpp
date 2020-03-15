//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "AutoSynthesizer.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  class AutoFixer : public RecursiveASTVisitor<AutoFixer> {
  private:
    Sema* m_Sema;
    DeclRefExpr* m_FoundDRE;
    llvm::DenseSet<NamedDecl*> m_HandledDecls;
  private:
  public:
    AutoFixer(Sema* S) : m_Sema(S), m_FoundDRE(0) {}

    void Fix(CompoundStmt* CS) {
      if (!CS->size())
        return;
      typedef llvm::SmallVector<Stmt*, 32> Statements;
      Statements Stmts;
      Stmts.append(CS->body_begin(), CS->body_end());
      for (Statements::iterator I = Stmts.begin(); I != Stmts.end(); ++I) {
        if (!TraverseStmt(*I) && !m_HandledDecls.count(m_FoundDRE->getDecl())) {
          Sema::DeclGroupPtrTy VDPtrTy
            = m_Sema->ConvertDeclToDeclGroup(m_FoundDRE->getDecl());
          StmtResult DS = m_Sema->ActOnDeclStmt(VDPtrTy,
                                                m_FoundDRE->getBeginLoc(),
                                                m_FoundDRE->getEndLoc());
          assert(!DS.isInvalid() && "Invalid DeclStmt.");
          I = Stmts.insert(I, DS.get());
          m_HandledDecls.insert(m_FoundDRE->getDecl());
        }
      }
      if (CS->size() != Stmts.size())
        CS->replaceStmts(m_Sema->getASTContext(), Stmts);
    }

    void Fix(CXXTryStmt* TS) {
      Fix(TS->getTryBlock());
      for(unsigned int h = 0; h < TS->getNumHandlers(); ++h) {
        Stmt *s = TS->getHandler(h)->getHandlerBlock();
        if (CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(s))
          Fix(CS);
        else if (CXXTryStmt *HandlerTS = dyn_cast_or_null<CXXTryStmt>(s))
          Fix(HandlerTS);
      }
    }

    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      const Decl* D = DRE->getDecl();
      if (const AnnotateAttr* A = D->getAttr<AnnotateAttr>())
        if (A->getAnnotation().equals("__Auto")) {
          m_FoundDRE = DRE;
          return false; // we abort on the first found candidate.
        }
      return true; // returning false will abort the in-depth traversal.
    }
  };
} // end namespace cling

namespace cling {
  AutoSynthesizer::AutoSynthesizer(clang::Sema* S)
    : ASTTransformer(S) {
    // TODO: We would like to keep that local without keeping track of all
    // decls that were handled in the AutoFixer. This can be done by removing
    // the __Auto attribute, but for now I am still hesitant to do it. Having
    // the __Auto attribute is very useful for debugging because it localize the
    // the problem if exists.
    m_AutoFixer.reset(new AutoFixer(S));
  }

  // pin the vtable here.
  AutoSynthesizer::~AutoSynthesizer()
  { }

  ASTTransformer::Result AutoSynthesizer::Transform(Decl* D) {
    if (FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
      // getBody() might return nullptr even though hasBody() is true for
      // late template parsed functions. We simply don't do auto auto on
      // those.
      Stmt *Body = FD->getBody();
      if (CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(Body))
        m_AutoFixer->Fix(CS);
      else if (CXXTryStmt *TS = dyn_cast_or_null<CXXTryStmt>(Body))
        m_AutoFixer->Fix(TS);
    }
    return Result(D, true);
  }
} // end namespace cling
