//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DynamicLookup.h"

#include "EnterUserCodeRAII.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/DynamicExprInfo.h"
#include "cling/Interpreter/Value.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Output.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Template.h"

#include "cling/Interpreter/DynamicLookupLifetimeHandler.h"

using namespace clang;

namespace {

  class StmtPrinterHelper : public PrinterHelper  {

  private:
    PrintingPolicy m_Policy;
    llvm::SmallVector<DeclRefExpr*, 4>& m_Addresses;
    Sema* m_Sema;
  public:

    StmtPrinterHelper(const PrintingPolicy& Policy,
                      llvm::SmallVector<DeclRefExpr*, 4>& Addresses,
                      Sema* S) :
      m_Policy(Policy), m_Addresses(Addresses), m_Sema(S) {}

    virtual ~StmtPrinterHelper() {}


    // Handle only DeclRefExprs since they are local and the call wrapper
    // won't "see" them. Consequently we don't need to handle:
    // * DependentScopeDeclRefExpr
    // * CallExpr
    // * MemberExpr
    // * CXXDependentScopeMemberExpr
    virtual bool handledStmt(Stmt* S, llvm::raw_ostream& OS) {
      if (DeclRefExpr* Node = dyn_cast<DeclRefExpr>(S))
        // Exclude the artificially dependent DeclRefExprs, created by the
        // Lookup
        if (!Node->isTypeDependent()) {
          if (NestedNameSpecifier* Qualifier = Node->getQualifier())
            Qualifier->print(OS, m_Policy);
          m_Addresses.push_back(Node);

          OS << "((";
          QualType T = Node->getType();
          if (!T->isArrayType())
            OS << '*';

          OS << '(';

          ASTContext &Ctx = m_Sema->getASTContext();

          OS << Ctx.getBaseElementType(T).getAsString();

          // We need to handle the arrays differently
          if (const ArrayType* AT = dyn_cast<ArrayType>(T.getTypePtr())) {
            OS << "(*)";
            T = AT->getElementType();

            while ((AT = dyn_cast<ArrayType>(T))) {
              // TODO: Fix other types of arrays
              if (const ConstantArrayType* CAT
                    = dyn_cast<ConstantArrayType>(AT))
                OS <<'[' << CAT->getSize().getZExtValue() << ']';


              T = AT->getElementType();
            }
          }
          else
            OS << '*';

          // end

          OS << ")@)";

          if (Node->hasExplicitTemplateArgs())
            printTemplateArgumentList(OS, Node->template_arguments(), m_Policy);
          if (Node->hasExplicitTemplateArgs())
            assert((Node->getTemplateArgs() || Node->getNumTemplateArgs()) && \
                   "There shouldn't be template paramlist");

          OS << ')';

          return true;
        }

      return false;
    }
  };

  class DynScopeDeclVisitor:
    public DeclVisitor<DynScopeDeclVisitor, bool> {
  private:
    cling::EvaluateTSynthesizer& m_EvalTSynth;

  public:
    DynScopeDeclVisitor(cling::EvaluateTSynthesizer& evalTSynth):
      m_EvalTSynth(evalTSynth) {}

    bool VisitFunctionDecl(FunctionDecl* FD) {
      if (Stmt* Body = FD->getBody()) {
        cling::ASTNodeInfo Replacement = m_EvalTSynth.Visit(Body);
        if (Replacement.hasErrorOccurred()) {
          FD->setBody(nullptr);
          return true;
        }

        if (Replacement.isForReplacement()) {
          // FIXME: support multiple Stmt Replacement!
          FD->setBody(Replacement.getAsSingleNode());
          return true;
        }
      }

      return false;
    }

    bool VisitDecl(Decl* D) {
      bool Replaced = false;
      if (auto DC = dyn_cast<DeclContext>(D)) {
        for (auto C: DC->decls())
          Replaced |= Visit(C);
      }
      return Replaced;
    }
  };
} // end anonymous namespace

namespace cling {

  // Constructors
  EvaluateTSynthesizer::EvaluateTSynthesizer(Sema* S)
    : ASTTransformer(S), m_EvalDecl(0), m_LifetimeHandlerDecl(0),
      m_LHgetMemoryDecl(0), m_DynamicExprInfoDecl(0), m_DeclContextDecl(0),
      m_gCling(0), m_CurDeclContext(0), m_Context(&S->getASTContext()),
      m_UniqueNameCounter(0), m_NestedCompoundStmts(0)
  { }

  // pin the vtable here.
  EvaluateTSynthesizer::~EvaluateTSynthesizer()
  { }

  void EvaluateTSynthesizer::Initialize() {
    // Most of the declaration we are looking up are in cling::runtime::internal
    NamespaceDecl* NSD = utils::Lookup::Namespace(m_Sema, "cling");
    NamespaceDecl* clingRuntimeNSD
      = utils::Lookup::Namespace(m_Sema, "runtime", NSD);
    NSD = utils::Lookup::Namespace(m_Sema, "internal", clingRuntimeNSD);

    // Find and set up EvaluateT
    DeclarationName Name = &m_Context->Idents.get("EvaluateT");

    LookupResult R(*m_Sema, Name, SourceLocation(), Sema::LookupOrdinaryName,
                     Sema::ForVisibleRedeclaration);
    assert(NSD && "There must be a valid namespace.");
    m_Sema->LookupQualifiedName(R, NSD);
    // We have specialized EvaluateT but we don't care because the templated
    // decl is needed.
    TemplateDecl* TplD = dyn_cast_or_null<TemplateDecl>(*R.begin());
    m_EvalDecl = dyn_cast<FunctionDecl>(TplD->getTemplatedDecl());
    assert(m_EvalDecl && "The Eval function not found!");

    // Find the LifetimeHandler declaration
    R.clear();
    Name = &m_Context->Idents.get("LifetimeHandler");
    R.setLookupName(Name);
    m_Sema->LookupQualifiedName(R, NSD);
    m_LifetimeHandlerDecl = R.getAsSingle<CXXRecordDecl>();
    assert(m_LifetimeHandlerDecl && "LifetimeHandler could not be found.");
    m_LifetimeHandlerDecl = m_LifetimeHandlerDecl->getDefinition();
    assert(m_LifetimeHandlerDecl && "LifetimeHandler is not defined");

    // Find the LifetimeHandler::getMemory declaration
    R.clear();
    Name = &m_Context->Idents.get("getMemory");
    R.setLookupName(Name);
    m_Sema->LookupQualifiedName(R, m_LifetimeHandlerDecl);
    m_LHgetMemoryDecl = R.getAsSingle<CXXMethodDecl>();
    assert(m_LHgetMemoryDecl && "LifetimeHandler::getMemory couldn't be found.");

    // Find the DynamicExprInfo declaration
    R.clear();
    Name = &m_Context->Idents.get("DynamicExprInfo");
    R.setLookupName(Name);
    m_Sema->LookupQualifiedName(R, NSD);
    m_DynamicExprInfoDecl = R.getAsSingle<CXXRecordDecl>();
    assert(m_DynamicExprInfoDecl && "DynExprInfo could not be found.");

    // Find the DeclContext declaration
    R.clear();
    Name = &m_Context->Idents.get("DeclContext");
    R.setLookupName(Name);
    NamespaceDecl* clangNSD = utils::Lookup::Namespace(m_Sema, "clang");
    m_Sema->LookupQualifiedName(R, clangNSD);
    m_DeclContextDecl = R.getAsSingle<CXXRecordDecl>();
    assert(m_DeclContextDecl && "clang::DeclContext decl could not be found.");

    // Find the gCling declaration
    R.clear();
    Name = &m_Context->Idents.get("gCling");
    R.setLookupName(Name);
    m_Sema->LookupQualifiedName(R, clingRuntimeNSD);
    m_gCling = R.getAsSingle<VarDecl>();
    assert(m_gCling && "gCling decl could not be found.");

    // Find and set the source locations to valid ones.
    R.clear();
    Name
      = &m_Context->Idents.get(
                               "InterpreterGeneratedCodeDiagnosticsMaybeIncorrect");
    R.setLookupName(Name);

    m_Sema->LookupQualifiedName(R, NSD);
    assert(!R.empty()
           && "Cannot find InterpreterGeneratedCodeDiagnosticsMaybeIncorrect");

    NamedDecl* ND = R.getFoundDecl();
    m_NoRange = ND->getSourceRange();
    m_NoSLoc = m_NoRange.getBegin();
    m_NoELoc = m_NoRange.getEnd();
  }

  ASTTransformer::Result EvaluateTSynthesizer::Transform(Decl* D) {
    if (!getCompilationOpts().DynamicScoping)
      return Result(D, true);

    if (FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
      if (FD->hasBody() && ShouldVisit(FD)) {
        // Find DynamicLookup specific builtins
        if (!m_EvalDecl)
          Initialize();

        // Set the decl context, which is needed by Evaluate.
        m_CurDeclContext = FD;
        ASTNodeInfo NewBody = Visit(D->getBody());
        if (NewBody.hasErrorOccurred()) {
          // Report unsupported feature.
          DiagnosticsEngine& Diags = m_Sema->getDiagnostics();
          unsigned diagID
          = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                  "Syntax error");
          Diags.Report(NewBody.getAsSingleNode()->getBeginLoc(), diagID);
          D->dump();
          if (NewBody.hasSingleNode())
            NewBody.getAs<Expr>()->dump();
          return Result(0, false); // Signal a fatal error.
        }
        FD->setBody(NewBody.getAsSingleNode());
      }
      assert ((!isa<BlockDecl>(D) || !isa<ObjCMethodDecl>(D))
              && "Not implemented yet!");
    }

    //TODO: Check for error before returning.
    return Result(D, true);
  }

  // StmtVisitor

  ASTNodeInfo EvaluateTSynthesizer::VisitStmt(Stmt* Node) {
    for (Stmt::child_iterator
           I = Node->child_begin(), E = Node->child_end(); I != E; ++I) {
      if (*I) {
        ASTNodeInfo NewNode = Visit(*I);
        assert(NewNode.hasSingleNode() &&
               "Cannot have more than one stmt at that point");

        if (NewNode.isForReplacement()) {
          if (Expr* E = NewNode.getAs<Expr>())
            // Assume void if still not escaped
            *I = SubstituteUnknownSymbol(m_Context->VoidTy, E);
        }
        else {
          *I = NewNode.getAsSingleNode();
        }
      }
    }

    return ASTNodeInfo(Node, 0);
  }

  // If the dynamic expression is in the conditional clause of the if
  // assume that the return type is bool, because we know that
  // everything in the condition of IfStmt is implicitly converted into bool
  ASTNodeInfo EvaluateTSynthesizer::VisitIfStmt(IfStmt* Node) {

    // See whether there is var defined. Eg: if (int i = f->call())
    // It will fall into DeclStmt.
    if (Node->getConditionVariableDeclStmt()) {
      // Removing the const, which shouldn't be dangerous
      VisitDeclStmt(const_cast<DeclStmt*>(
                                         Node->getConditionVariableDeclStmt()));
    }

    // Handle the case where the dynamic expression is in the condition of the
    // stmt.
    ASTNodeInfo IfCondInfo = Visit(Node->getCond());
    if (IfCondInfo.isForReplacement())
      if (Expr* IfCondExpr = IfCondInfo.getAs<Expr>()) {
        Node->setCond(SubstituteUnknownSymbol(m_Context->BoolTy, IfCondExpr));
        //return ASTNodeInfo(Node, /*needs eval*/false);
      }

    // Visit the other parts - they will fall naturally into CompoundStmt
    // where we know what to do. For Stmt, though, we need to substitute here,
    // knowing the "target" type.
    ASTNodeInfo thenInfo = Visit(Node->getThen());
    if (thenInfo.isForReplacement())
      Node->setThen(SubstituteUnknownSymbol(m_Context->VoidTy,
                                            thenInfo.getAs<Expr>()));
    if (Stmt* ElseExpr = Node->getElse()) {
      ASTNodeInfo elseInfo = Visit(ElseExpr);
      if (elseInfo.isForReplacement())
        Node->setElse(SubstituteUnknownSymbol(m_Context->VoidTy,
                                              elseInfo.getAs<Expr>()));
    }

    return ASTNodeInfo(Node, false);
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitCompoundStmt(CompoundStmt* Node) {
    ++m_NestedCompoundStmts;
    ASTNodes Children;
    ASTNodes NewChildren;
    if (GetChildren(Children, Node)) {
      ASTNodes::iterator it;
      for (it = Children.begin(); it != Children.end(); ++it) {
        ASTNodeInfo NewNode = Visit(*it);
        if (NewNode.hasErrorOccurred())
          return NewNode; // abort.
        if (!NewNode.hasSingleNode()) {

          ASTNodes& NewStmts(NewNode.getNodes());
          for(unsigned i = 0; i < NewStmts.size(); ++i)
            NewChildren.push_back(NewStmts[i]);
        }
        else {
          if (NewNode.isForReplacement()) {
            if (Expr* E = NewNode.getAs<Expr>()) {
              // Check whether value printer has been requested
              bool valuePrinterReq = false;
              // If this was the last or the last is not null stmt, means that
              // we need to value print.
              // If this is in a wrapper function's body then look for VP.
              if (FunctionDecl* FD = dyn_cast<FunctionDecl>(m_CurDeclContext))
                valuePrinterReq
                  = m_NestedCompoundStmts < 2  && utils::Analyze::IsWrapper(FD)
                  && ((it+1) == Children.end() || ((it+2) == Children.end()
                                                   && !isa<NullStmt>(*(it+1))));

              // Assume void if still not escaped
              NewChildren.push_back(SubstituteUnknownSymbol(m_Context->VoidTy,E,
                                                            valuePrinterReq));
            }
          }
          else
            NewChildren.push_back(*it);
        }
      }
    }

    Node->replaceStmts(*m_Context, NewChildren);

    --m_NestedCompoundStmts;
    return ASTNodeInfo(Node, 0);
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitDeclStmt(DeclStmt* Node) {
    // Visit all the children, which are the contents of the DeclGroupRef
    if (VarDecl* CuredDecl = dyn_cast<VarDecl>(Node->getSingleDecl())) {
      //FIXME: don't assume there is only one decl.
      assert(Node->isSingleDecl() && "There is more that one decl in stmt");
      for (Stmt::child_iterator
             I = Node->child_begin(), E = Node->child_end(); I != E; ++I) {
        Expr* InitExpr = cast_or_null<Expr>(*I);
        if (!InitExpr || !IsArtificiallyDependent(InitExpr))
          continue;

        QualType CuredDeclTy = CuredDecl->getType();
        if (isa<AutoType>(CuredDeclTy)) {
          ASTNodeInfo result(Node, false);
          result.setErrorOccurred();
          return result;
        }
        // check if the case is sometype * somevar = init;
        // or some_builtin_type somevar = init;
        if (CuredDecl->hasInit() &&
            (CuredDeclTy->isAnyPointerType() || !CuredDeclTy->isRecordType())) {
          *I = SubstituteUnknownSymbol(CuredDeclTy, CuredDecl->getInit());
          continue;
        }

        // 1. Check whether this is the case of MyClass A(dep->symbol())
        // 2. Insert the RuntimeUniverse's LifetimeHandler instance
        // 3. Change the A's initializer to *(MyClass*)instance.getMemory()
        // 4. Make A reference (&A)
        // 5. Set the new initializer of A
        if (CuredDeclTy->isLValueReferenceType())
          continue;

        // Set Sema's Current DeclContext to the one we need
        DeclContext* OldDC = m_Sema->CurContext;
        m_Sema->CurContext = CuredDecl->getDeclContext();

        ASTNodeInfo NewNode;
        // 2.1 Get unique name for the LifetimeHandler instance and
        // initialize it
        std::string UniqueName = createUniqueName();
        IdentifierInfo& II = m_Context->Idents.get(UniqueName);

        // Prepare the initialization Exprs.
        // We want to call LifetimeHandler(DynamicExprInfo* ExprInfo,
        //                                 DeclContext DC,
        //                                 const char* type)
        //                                 Interpreter* interp)
        llvm::SmallVector<Expr*, 4> Inits;
        // Add MyClass in LifetimeHandler unique(DynamicExprInfo* ExprInfo
        //                                       DC,
        //                                       "MyClass"
        //                                       Interpreter* Interp)
        // Build Arg0 DynamicExprInfo
        Inits.push_back(BuildDynamicExprInfo(InitExpr));
        // Build Arg1 DeclContext* DC
        QualType DCTy = m_Context->getTypeDeclType(m_DeclContextDecl);
        Inits.push_back(utils::Synthesize::CStyleCastPtrExpr(m_Sema, DCTy,
                                                     (uintptr_t)m_CurDeclContext)
                        );
        // Build Arg2 llvm::StringRef
        // Get the type of the type without specifiers
        PrintingPolicy Policy(m_Context->getLangOpts());
        Policy.SuppressTagKeyword = 1;
        std::string Res;
        CuredDeclTy.getAsStringInternal(Res, Policy);
        Inits.push_back(ConstructConstCharPtrExpr(Res.c_str()));

        // Build Arg3 cling::Interpreter
        CXXScopeSpec CXXSS;
        DeclarationNameInfo NameInfo(m_gCling->getDeclName(),
                                     m_gCling->getBeginLoc());
        Expr* gClingDRE
          = m_Sema->BuildDeclarationNameExpr(CXXSS, NameInfo ,m_gCling).get();
        Inits.push_back(gClingDRE);

        // 2.3 Create a variable from LifetimeHandler.
        QualType HandlerTy = m_Context->getTypeDeclType(m_LifetimeHandlerDecl);
        TypeSourceInfo* TSI = m_Context->getTrivialTypeSourceInfo(HandlerTy,
                                                                  m_NoSLoc);
        VarDecl* HandlerInstance = VarDecl::Create(*m_Context,
                                                   CuredDecl->getDeclContext(),
                                                   m_NoSLoc,
                                                   m_NoSLoc,
                                                   &II,
                                                   HandlerTy,
                                                   TSI,
                                                   SC_None);

        // 2.4 Call the best-match constructor. The method does overload
        // resolution of the constructors and then initializes the new
        // variable with it
        ExprResult InitExprResult
          = m_Sema->ActOnParenListExpr(m_NoSLoc,
                                       m_NoELoc,
                                       Inits);
        m_Sema->AddInitializerToDecl(HandlerInstance,
                                     InitExprResult.get(),
                                     /*DirectInit*/ true);

        // 2.5 Register the instance in the enclosing context
        CuredDecl->getDeclContext()->addDecl(HandlerInstance);
        NewNode.addNode(new (m_Context)
                        DeclStmt(DeclGroupRef(HandlerInstance),
                                 m_NoSLoc,
                                 m_NoELoc)
                        );

        // 3.1 Build a DeclRefExpr, which holds the object
        DeclRefExpr* MemberExprBase
          = m_Sema->BuildDeclRefExpr(HandlerInstance,
                                     HandlerTy,
                                     VK_LValue,
                                     m_NoSLoc
                                     );
        // 3.2 Create a MemberExpr to getMemory from its declaration.
        CXXScopeSpec SS;
        LookupResult MemberLookup(*m_Sema, m_LHgetMemoryDecl->getDeclName(),
                                  m_NoSLoc, Sema::LookupMemberName);
        // Add the declaration as if doesn't exist.
        // TODO: Check whether this is the most appropriate variant
        MemberLookup.addDecl(m_LHgetMemoryDecl, AS_public);
        MemberLookup.resolveKind();
        Expr* MemberExpr
           = m_Sema->BuildMemberReferenceExpr(MemberExprBase,
                                              HandlerTy,
                                              m_NoSLoc,
                                              /*IsArrow=*/false,
                                              SS,
                                              m_NoSLoc,
                                              /*FirstQualifierInScope=*/0,
                                              MemberLookup,
                                              /*TemplateArgs=*/0,
                                              /*Scope*/nullptr).get();
        // 3.3 Build the actual call
        Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
        Expr* theCall = m_Sema->ActOnCallExpr(S,
                                              MemberExpr,
                                              m_NoSLoc,
                                              MultiExprArg(),
                                              m_NoELoc).get();
        // Cast to the type LHS type
        Expr* Result
          = utils::Synthesize::CStyleCastPtrExpr(m_Sema, CuredDeclTy, theCall);
        // Cast once more (dereference the cstyle cast)
        Result = m_Sema->BuildUnaryOp(S, m_NoSLoc, UO_Deref, Result).get();
        // 4.
        CuredDecl->setType(m_Context->getLValueReferenceType(CuredDeclTy));
        // 5.
        CuredDecl->setInit(Result);

        NewNode.addNode(Node);

        // Restore Sema's original DeclContext
        m_Sema->CurContext = OldDC;
        // FIXME: this only works if this is the only Decl in the DeclGroup...
        return NewNode;
      }
    }

    // Recurse over Decls; they might need transformations, too; e.g. in
    // void wrapper() { struct X {  void f() { ++dynScope; } }; }
    bool HaveReplacement = false;
    DynScopeDeclVisitor DSDV(*this);
    // DynScopeDeclVisitor always changes the Decls in-place, no need to pick up
    // new ones.
    for (auto DGI: Node->getDeclGroup())
      HaveReplacement |= DSDV.Visit(DGI);
    return ASTNodeInfo(Node, HaveReplacement);
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitCXXDeleteExpr(CXXDeleteExpr* Node) {
    ASTNodeInfo deleteArg = Visit(Node->getArgument());
    return ASTNodeInfo(Node, /*needs eval*/deleteArg.isForReplacement());
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitCXXDependentScopeMemberExpr(
                                            CXXDependentScopeMemberExpr* Node) {
    ASTNodeInfo ContentTransform = Visit(Node->getBase());
    // Simply skip the CXXDependentScopeMemberExpr and only use its content.
    assert(ContentTransform.hasSingleNode() &&
           "Cannot substitute multiple nodes for CXXDependentScopeMemberExpr.");
    if (ContentTransform.isForReplacement()) {
      Expr* Replacement = ContentTransform.getAs<Expr>();
      return ASTNodeInfo(Replacement, ContentTransform.isForReplacement());
    }
    ContentTransform.setErrorOccurred();
    return ContentTransform;
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitExpr(Expr* Node) {
    bool NeedsEval = false;
    for (Stmt::child_iterator
           I = Node->child_begin(), E = Node->child_end(); I != E; ++I) {
      if (*I) {
        ASTNodeInfo NewNode = Visit(*I);
        assert(NewNode.hasSingleNode() &&
               "Cannot have more than one stmt at that point");
        if (NewNode.isForReplacement())
          NeedsEval = true;
        *I = NewNode.getAsSingleNode();
      }
    }
    return ASTNodeInfo(Node, NeedsEval);
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitBinaryOperator(BinaryOperator* Node) {
    ASTNodeInfo rhs = Visit(Node->getRHS());
    ASTNodeInfo lhs;

    lhs = Visit(Node->getLHS());

    assert((lhs.hasSingleNode() || rhs.hasSingleNode()) &&
           "1:N replacements are not implemented yet!");

    // Try find out the type of the left-hand-side of the operator
    // and give the hint to the right-hand-side in order to replace the
    // dependent symbol
    if (Node->isAssignmentOp() && rhs.isForReplacement() &&
        !lhs.isForReplacement()) {
      if (Expr* LHSExpr = lhs.getAs<Expr>())
        if (!IsArtificiallyDependent(LHSExpr)) {
          const QualType LHSTy = LHSExpr->getType();
          Node->setRHS(SubstituteUnknownSymbol(LHSTy, rhs.castTo<Expr>()));
          Node->setTypeDependent(false);
          Node->setValueDependent(false);
          return ASTNodeInfo(Node, /*needs eval*/false);
        }
    }

    return ASTNodeInfo(Node, IsArtificiallyDependent(Node));
  }


  ASTNodeInfo
  EvaluateTSynthesizer::VisitArraySubscriptExpr(ArraySubscriptExpr* Node) {
    ASTNodeInfo rhs = Visit(Node->getRHS());
    ASTNodeInfo lhs;

    lhs = Visit(Node->getLHS());

    assert((lhs.hasSingleNode() || rhs.hasSingleNode()) &&
           "1:N replacements are not implemented yet!");

    return ASTNodeInfo(Node, IsArtificiallyDependent(Node));
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitCallExpr(CallExpr* E) {
    // FIXME: Maybe we need to handle the arguments
    //ASTNodeInfo NewNode = Visit(E->getCallee());
    return ASTNodeInfo (E, IsArtificiallyDependent(E));
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitDeclRefExpr(DeclRefExpr* DRE) {
    return ASTNodeInfo(DRE, IsArtificiallyDependent(DRE));
  }

  ASTNodeInfo EvaluateTSynthesizer::VisitDependentScopeDeclRefExpr(
                                              DependentScopeDeclRefExpr* Node) {
    return ASTNodeInfo(Node, IsArtificiallyDependent(Node));
  }

  // end StmtVisitor

  // EvalBuilder

  Expr* EvaluateTSynthesizer::SubstituteUnknownSymbol(const QualType InstTy,
                                                      Expr* SubTree,
                                                      bool ValuePrinterReq) {
    assert(SubTree && "No subtree specified!");

    //Build the arguments for the call
    llvm::SmallVector<Expr*, 2> CallArgs;

    // Build Arg0
    Expr* Arg0 = BuildDynamicExprInfo(SubTree, ValuePrinterReq);
    CallArgs.push_back(Arg0);

    // Build Arg1
    QualType DCTy = m_Context->getTypeDeclType(m_DeclContextDecl);
    Expr* Arg1 = utils::Synthesize::CStyleCastPtrExpr(m_Sema, DCTy,
                                                    (uintptr_t)m_CurDeclContext);
    CallArgs.push_back(Arg1);

    // Build the call
    assert(Arg0 && Arg1 && "Arguments missing!");
    CallExpr* EvalCall = BuildEvalCallExpr(InstTy, SubTree, CallArgs);

    // Add substitution mapping
    getSubstSymbolMap()[EvalCall] = SubTree;

    return EvalCall;
  }

  Expr* EvaluateTSynthesizer::BuildDynamicExprInfo(Expr* SubTree,
                                                   bool ValuePrinterReq) {
    // We need to evaluate it in its own context. Evaluation on the global
    // scope per se can break for example the compound literals, which have
    // to be constants (see [C99 6.5.2.5])
    Sema::ContextRAII pushedDC(*m_Sema, m_CurDeclContext);

    // 1. Get the expression containing @-s and get the variable addresses
    llvm::SmallVector<DeclRefExpr*, 4> Addresses;
    ostrstream OS;
    const PrintingPolicy& Policy = m_Context->getPrintingPolicy();

    StmtPrinterHelper helper(Policy, Addresses, m_Sema);
    // In case when we print non paren inits like int i = h->Draw();
    // not int i(h->Draw()). This simplifies the LifetimeHandler's
    // constructor, there we don't need to add parenthesis while
    // wrapping the expression.
    if (!isa<ParenListExpr>(SubTree))
      OS << '(';
    SubTree->printPretty(OS, &helper, Policy);
    if (!isa<ParenListExpr>(SubTree))
      OS << ')';

    // 2. Build the template
    Expr* ExprTemplate = ConstructConstCharPtrExpr(OS.str());

    // 3. Build the array of addresses
    QualType VarAddrTy = m_Sema->BuildArrayType(m_Context->VoidPtrTy,
                                                ArrayType::Normal,
                                                /*ArraySize*/0,
                                                /*IndexTypeQuals*/0,
                                                m_NoRange,
                                                DeclarationName() );

    llvm::SmallVector<Expr*, 2> Inits;
    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    for (unsigned int i = 0; i < Addresses.size(); ++i) {

      Expr* UnOp
        = m_Sema->BuildUnaryOp(S, Addresses[i]->getBeginLoc(), UO_AddrOf,
                               Addresses[i]).get();
      if (!UnOp) {
        // Not good, return what we had.
        cling::errs() << "Error while creating dynamic expression for:\n  ";
        SubTree->printPretty(cling::errs(), 0 /*PrinterHelper*/,
                             m_Context->getPrintingPolicy(), 2);
        cling::errs() << "\n";
#ifndef NDEBUG
        cling::errs() <<
          "with internal representation (look for <dependent type>):\n";
        SubTree->dump(cling::errs(), m_Sema->getSourceManager());
#endif
        return SubTree;
      }
      m_Sema->ImpCastExprToType(UnOp,
                                m_Context->getPointerType(m_Context->VoidPtrTy),
                                CK_BitCast);
      Inits.push_back(UnOp);
    }

    // We need valid source locations to avoid assert(InitList.isExplicit()...)
    InitListExpr* ILE = m_Sema->ActOnInitList(m_NoSLoc,
                                              Inits,
                                              m_NoELoc).getAs<InitListExpr>();
    TypeSourceInfo* TSI
      = m_Context->getTrivialTypeSourceInfo(VarAddrTy, m_NoSLoc);
    Expr* ExprAddresses = m_Sema->BuildCompoundLiteralExpr(m_NoSLoc,
                                                           TSI,
                                                           m_NoELoc,
                                                           ILE).get();
    assert (ExprAddresses && "Could not build the void* array");
    if (!ExprAddresses)
       return SubTree;

    m_Sema->ImpCastExprToType(ExprAddresses,
                              m_Context->getPointerType(m_Context->VoidPtrTy),
                              CK_ArrayToPointerDecay);

    // Is the result of the expression to be printed or not
    Expr* VPReq = 0;
    if (ValuePrinterReq)
      VPReq = m_Sema->ActOnCXXBoolLiteral(m_NoSLoc, tok::kw_true).get();
    else
      VPReq = m_Sema->ActOnCXXBoolLiteral(m_NoSLoc, tok::kw_false).get();

    llvm::SmallVector<Expr*, 4> CtorArgs;
    CtorArgs.push_back(ExprTemplate);
    CtorArgs.push_back(ExprAddresses);
    CtorArgs.push_back(VPReq);

    // 4. Call the constructor
    QualType ExprInfoTy = m_Context->getTypeDeclType(m_DynamicExprInfoDecl);
    ExprResult Initializer = m_Sema->ActOnParenListExpr(m_NoSLoc, m_NoELoc,
                                                        CtorArgs);
    TypeSourceInfo* TrivialTSI
      = m_Context->getTrivialTypeSourceInfo(ExprInfoTy, SourceLocation());
    Expr* Result = m_Sema->BuildCXXNew(m_NoSLoc,
                                       /*UseGlobal=*/false,
                                       m_NoSLoc,
                                       /*PlacementArgs=*/MultiExprArg(),
                                       m_NoELoc,
                                       m_NoRange,
                                       ExprInfoTy,
                                       TrivialTSI,
                                       /*ArraySize=*/{},
                                       //BuildCXXNew depends on the SLoc to be
                                       //valid!
                                       // TODO: Propose a patch in clang
                                       m_NoRange,
                                       Initializer.get()
                                       ).get();
    return Result;
  }

  Expr* EvaluateTSynthesizer::ConstructConstCharPtrExpr(llvm::StringRef Value) {
    const QualType CChar = m_Context->CharTy.withConst();

    unsigned bitSize = m_Context->getTypeSize(m_Context->VoidPtrTy);
    llvm::APInt ArraySize(bitSize, Value.size() + 1);
    const QualType CCArray = m_Context->getConstantArrayType(CChar,
                                                            ArraySize,
                                                            ArrayType::Normal,
                                                          /*IndexTypeQuals=*/0);

    StringLiteral::StringKind Kind = StringLiteral::Ascii;
    Expr* Result = StringLiteral::Create(*m_Context,
                                         Value,
                                         Kind,
                                         /*Pascal=*/false,
                                         CCArray,
                                         m_NoSLoc);
    m_Sema->ImpCastExprToType(Result,
                              m_Context->getPointerType(CChar),
                              CK_ArrayToPointerDecay);

    return Result;
  }


  // Here is the test Eval function specialization. Here the CallExpr to the
  // function is created.
  CallExpr*
  EvaluateTSynthesizer::BuildEvalCallExpr(const QualType InstTy,
                                          Expr* SubTree,
                                        llvm::SmallVector<Expr*, 2>& CallArgs) {
    // Set up new context for the new FunctionDecl
    DeclContext* PrevContext = m_Sema->CurContext;

    m_Sema->CurContext = m_EvalDecl->getDeclContext();

    // Create template arguments
    Sema::InstantiatingTemplate Inst(*m_Sema, m_NoSLoc, m_EvalDecl);
    // Before instantiation we need the canonical type
    TemplateArgument Arg(InstTy.getCanonicalType());
    TemplateArgumentList TemplateArgs(TemplateArgumentList::OnStack, Arg);

    // Substitute the declaration of the templated function, with the
    // specified template argument
    Decl* D = m_Sema->SubstDecl(m_EvalDecl,
                                m_EvalDecl->getDeclContext(),
                                MultiLevelTemplateArgumentList(TemplateArgs));

    FunctionDecl* Fn = dyn_cast<FunctionDecl>(D);

    // IncrementalParser is opening a Transaction for the transformations which
    // will happily capture these decls.
#if 0
    // We expect incoming declarations (instantiations) and we
    // need to open the transaction to collect them.
    Transaction::State oldState = getTransaction()->getState();
    getTransaction()->setState(Transaction::kCollecting);
#endif

    // Creates new body of the substituted declaration
    m_Sema->InstantiateFunctionDefinition(Fn->getLocation(), Fn, true, true);

    m_Sema->CurContext = PrevContext;

    const FunctionProtoType* FPT = Fn->getType()->getAs<FunctionProtoType>();
    FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
    llvm::ArrayRef<QualType> ArgTypes(FPT->param_type_begin(),
                                      FPT->getNumParams());
    QualType FnTy = m_Context->getFunctionType(Fn->getReturnType(),
                                               ArgTypes,
                                               EPI);
    DeclRefExpr* DRE = m_Sema->BuildDeclRefExpr(Fn,
                                                FnTy,
                                                VK_RValue,
                                                m_NoSLoc
                                                );
#if 0
    getTransaction()->setState(oldState);
#endif

    // TODO: Figure out a way to avoid passing in wrong source locations
    // of the symbol being replaced. This is important when we calculate the
    // size of the memory buffers and may lead to creation of wrong wrappers.
    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    CallExpr* EvalCall = m_Sema->ActOnCallExpr(S,
                                               DRE,
                                               SubTree->getBeginLoc(),
                                               CallArgs,
                                               SubTree->getEndLoc()
                                               ).getAs<CallExpr>();
    assert (EvalCall && "Cannot create call to Eval");

    return EvalCall;

  }

  // end EvalBuilder

  bool EvaluateTSynthesizer::ShouldVisit(FunctionDecl* D) {
    // FIXME: Here we should have our custom attribute.
    if (AnnotateAttr* A = D->getAttr<AnnotateAttr>())
      if (A->getAnnotation().equals("__ResolveAtRuntime"))
        return true;
    return false;
  }

  bool EvaluateTSynthesizer::IsArtificiallyDependent(Expr* Node) {
    if (!Node->isValueDependent() && !Node->isTypeDependent()
        && !Node->isInstantiationDependent())
      return false;
    DeclContext* DC = m_CurDeclContext;
    while (DC) {
      if (DC->isDependentContext())
        return false;
      DC = DC->getParent();
    }
    return true;
  }

  bool EvaluateTSynthesizer::GetChildren(ASTNodes& Children, Stmt* Node) {
    if (std::distance(Node->child_begin(), Node->child_end()) < 1)
      return false;
    for (Stmt::child_iterator
           I = Node->child_begin(), E = Node->child_end(); I != E; ++I) {
      Children.push_back(*I);
    }
    return true;
  }

  std::string EvaluateTSynthesizer::createUniqueName() {
    stdstrstream Strm;
    Strm << "__dynamic" << utils::Synthesize::UniquePrefix
         << m_UniqueNameCounter++;
    return Strm.str();
  }
  

  // end Helpers

  namespace runtime {
  namespace internal {
    // Implementations from DynamicLookupLifetimeHandler.h
    LifetimeHandler::LifetimeHandler(DynamicExprInfo* ExprInfo,
                                     clang::DeclContext* DC,
                                     const char* type,
                                     Interpreter* Interp):
      m_Interpreter(Interp), m_Type(type) {
      std::string ctor("new ");
      ctor += type;
      ctor += ExprInfo->getExpr();
      LockCompilationDuringUserCodeExecutionRAII LCDUCER(*Interp);
      Value res = Interp->Evaluate(ctor.c_str(), DC,
                                   ExprInfo->isValuePrinterRequested()
                                   );
      m_Memory = (void*)res.getAs<void*>();
    }

    LifetimeHandler::~LifetimeHandler() {
      ostrstream stream;
      stream << "delete (" << m_Type << "*) " << m_Memory << ";";
      LockCompilationDuringUserCodeExecutionRAII LCDUCER(*m_Interpreter);
      m_Interpreter->execute(stream.str());
    }
  } // end namespace internal
  } // end namespace runtime
} // end namespace cling
