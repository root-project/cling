//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ValuePrinterSynthesizer.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/raw_os_ostream.h"

#include <iostream>

using namespace clang;

namespace cling {

  ValuePrinterSynthesizer::ValuePrinterSynthesizer(clang::Sema* S, 
                                                   llvm::raw_ostream* Stream)
    : TransactionTransformer(S), m_Context(&S->getASTContext()) {
    if (Stream)
      m_ValuePrinterStream.reset(Stream);
    else 
      m_ValuePrinterStream.reset(new llvm::raw_os_ostream(std::cout));
  }


  // pin the vtable here.
  ValuePrinterSynthesizer::~ValuePrinterSynthesizer()
  { }

  void ValuePrinterSynthesizer::Transform() {
    if (getTransaction()->getCompilationOpts().ValuePrinting 
        == CompilationOptions::VPDisabled)
      return;

    for (Transaction::const_iterator I = getTransaction()->decls_begin(), 
           E = getTransaction()->decls_end(); I != E; ++I)
      if(!tryAttachVP(*I))
        return setTransaction(0); // On error set to NULL.
  }

  bool ValuePrinterSynthesizer::tryAttachVP(DeclGroupRef DGR) {
    for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end(); I != E; ++I)
      if (FunctionDecl* FD = dyn_cast<FunctionDecl>(*I)) {
        if (!utils::Analyze::IsWrapper(FD))
          continue;
        const CompilationOptions& CO(getTransaction()->getCompilationOpts());
        if (CO.ValuePrinting == CompilationOptions::VPDisabled)
          return true; // Nothing to do.

        // We have to be able to mark the expression for printout. There are
        // three scenarios:
        // 0: Expression printing disabled - don't do anything just exit.
        // 1: Expression printing enabled - print no matter what.
        // 2: Expression printing auto - analyze - rely on the omitted ';' to
        //    not produce the suppress marker.
        int indexOfLastExpr = -1;
        if (Expr* To = utils::Analyze::GetLastExpr(FD, &indexOfLastExpr)) {
          // Update the CompoundStmt body, avoiding alloc/dealloc of all the el.
          CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
          assert(CS && "Missing body?");

          switch (CO.ValuePrinting) {
          case CompilationOptions::VPDisabled:
            assert("Don't wait that long. Exit early!");
            break;
          case CompilationOptions::VPEnabled:
            break;
          case CompilationOptions::VPAuto:
            if ((int)CS->size() > indexOfLastExpr+1 
                && (*(CS->body_begin() + indexOfLastExpr + 1))
                && isa<NullStmt>(*(CS->body_begin() + indexOfLastExpr + 1)))
              return true; // If next is NullStmt disable VP is disabled - exit.
            break;
          }

          // We can't PushDeclContext, because we don't have scope.
          Sema::ContextRAII pushedDC(*m_Sema, FD);

          if (To) {
            // Strip the parenthesis if any
            if (ParenExpr* PE = dyn_cast<ParenExpr>(To))
              To = PE->getSubExpr();
            
            Expr* Result = 0;
            if (m_Sema->getLangOpts().CPlusPlus)
              Result = SynthesizeCppVP(To);
            else
              Result = SynthesizeVP(To);

            if (Result)
              *(CS->body_begin()+indexOfLastExpr) = Result;
          }
          // Clear the artificial NullStmt-s
          if (!ClearNullStmts(CS)) {
            // FIXME: Why it is here? Shouldn't it be in DeclExtractor?
            // if no body remove the wrapper
            DeclContext* DC = FD->getDeclContext();
            Scope* S = m_Sema->getScopeForContext(DC);
            if (S)
              S->RemoveDecl(FD);
            DC->removeDecl(FD);
          }
        }
      }

    return true;
  }

  // We need to artificially create:
  // cling::valuePrinterInternal::PrintValue((void*) raw_ostream,
  //                                         (ASTContext)Ctx, (Expr*)E, &i);
  Expr* ValuePrinterSynthesizer::SynthesizeCppVP(Expr* E) {
    QualType QT = E->getType();
    // For now we skip void and function pointer types.
    if (!QT.isNull() && (QT->isVoidType() || QT->isFunctionType()))
      return 0;

    // 1. Call gCling->getValuePrinterStream()
    // 1.1. Find gCling
    SourceLocation NoSLoc = SourceLocation();

    NamespaceDecl* NSD = utils::Lookup::Namespace(m_Sema, "cling");
    NSD = utils::Lookup::Namespace(m_Sema, "valuePrinterInternal", NSD);


    DeclarationName PVName = &m_Context->Idents.get("PrintValue");
    LookupResult R(*m_Sema, PVName, NoSLoc, Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);
    assert(NSD && "There must be a valid namespace.");
    m_Sema->LookupQualifiedName(R, NSD);
    assert(!R.empty() && "Cannot find PrintValue(...)");

    CXXScopeSpec CSS;
    Expr* UnresolvedLookup
      = m_Sema->BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).take();

    // 2.4. Prepare the params

    // 2.4.1 Lookup the llvm::raw_ostream
    CXXRecordDecl* RawOStreamRD
      = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(m_Sema, "raw_ostream",
                                                 utils::Lookup::Namespace(m_Sema,
                                                                "llvm")));

    assert(RawOStreamRD && "Declaration of the expr not found!");
    QualType RawOStreamRDTy = m_Context->getTypeDeclType(RawOStreamRD);
    // 2.4.2 Lookup the expr type
    CXXRecordDecl* ExprRD
      = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(m_Sema, "Expr",
                                                 utils::Lookup::Namespace(m_Sema,
                                                                "clang")));
   assert(ExprRD && "Declaration of the expr not found!");
    QualType ExprRDTy = m_Context->getTypeDeclType(ExprRD);
    // 2.4.3 Lookup ASTContext type
    CXXRecordDecl* ASTContextRD
      = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(m_Sema, "ASTContext",
                                                 utils::Lookup::Namespace(m_Sema,
                                                                "clang")));
    assert(ASTContextRD && "Declaration of the expr not found!");
    QualType ASTContextRDTy = m_Context->getTypeDeclType(ASTContextRD);

    Expr* RawOStreamTy
      = utils::Synthesize::CStyleCastPtrExpr(m_Sema, RawOStreamRDTy,
                                             (uint64_t)m_ValuePrinterStream.get()
                                             );

    Expr* ExprTy = utils::Synthesize::CStyleCastPtrExpr(m_Sema, ExprRDTy, 
                                                        (uint64_t)E);
    Expr* ASTContextTy 
      = utils::Synthesize::CStyleCastPtrExpr(m_Sema, ASTContextRDTy,
                                             (uint64_t)m_Context);

    // E might contain temporaries. This means that the topmost expr is
    // ExprWithCleanups. This contains the information about the temporaries and
    // signals when they should be destroyed.
    // Here we replace E with call to value printer and we must extend the life
    // time of those temporaries to the end of the new CallExpr.
    bool NeedsCleanup = false;
    if (ExprWithCleanups* EWC = dyn_cast<ExprWithCleanups>(E)) {
      E = EWC->getSubExpr();
      NeedsCleanup = true;
    }


    llvm::SmallVector<Expr*, 4> CallArgs;
    CallArgs.push_back(RawOStreamTy);
    CallArgs.push_back(ExprTy);
    CallArgs.push_back(ASTContextTy);
    CallArgs.push_back(E);

    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    Expr* Result = m_Sema->ActOnCallExpr(S, UnresolvedLookup, NoSLoc,
                                         CallArgs, NoSLoc).take();

    Result = m_Sema->ActOnFinishFullExpr(Result).take();
    if (NeedsCleanup && !isa<ExprWithCleanups>(Result)) {
      llvm::ArrayRef<ExprWithCleanups::CleanupObject> Cleanups;
      ExprWithCleanups* EWC
        = ExprWithCleanups::Create(*m_Context, Result, Cleanups);
      Result = EWC;
    }

    assert(Result && "Cannot create value printer!");

    return Result;
  }

  // We need to artificially create:
  // cling_PrintValue(void* (ASTContext)C, void* (Expr)E, const void* (&i)
  Expr* ValuePrinterSynthesizer::SynthesizeVP(Expr* E) {
    QualType QT = E->getType();
    // For now we skip void and function pointer types.
    if (!QT.isNull() && (QT->isVoidType() || QT->isFunctionType()))
      return 0;

    // Find cling_PrintValue
    SourceLocation NoSLoc = SourceLocation();
    DeclarationName PVName = &m_Context->Idents.get("cling_PrintValue");
    LookupResult R(*m_Sema, PVName, NoSLoc, Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);

    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    m_Sema->LookupName(R, S);
    assert(!R.empty() && "Cannot find PrintValue(...)");

    CXXScopeSpec CSS;
    Expr* UnresolvedLookup
      = m_Sema->BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).take();


    Expr* VoidEArg = utils::Synthesize::CStyleCastPtrExpr(m_Sema, 
                                                          m_Context->VoidPtrTy,
                                                          (uint64_t)E);
    Expr* VoidCArg = utils::Synthesize::CStyleCastPtrExpr(m_Sema, 
                                                          m_Context->VoidPtrTy,
                                                          (uint64_t)m_Context);

    if (!QT->isPointerType()) {
      while(ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(E))
        E = ICE->getSubExpr();
      E = m_Sema->BuildUnaryOp(S, NoSLoc, UO_AddrOf, E).take();
    }

    llvm::SmallVector<Expr*, 4> CallArgs;
    CallArgs.push_back(VoidEArg);
    CallArgs.push_back(VoidCArg);
    CallArgs.push_back(E);

    Expr* Result = m_Sema->ActOnCallExpr(S, UnresolvedLookup, NoSLoc,
                                         CallArgs, NoSLoc).take();
    assert(Result && "Cannot create value printer!");

    return Result;
  }


  unsigned ValuePrinterSynthesizer::ClearNullStmts(CompoundStmt* CS) {
    llvm::SmallVector<Stmt*, 8> FBody;
    for (StmtRange range = CS->children(); range; ++range)
      if (!isa<NullStmt>(*range))
        FBody.push_back(*range);

    CS->setStmts(*m_Context, FBody.data(), FBody.size());
    return FBody.size();
  }

} // namespace cling
