//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DYNAMIC_LOOKUP_H
#define CLING_DYNAMIC_LOOKUP_H

#include "ASTTransformer.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Ownership.h"

#include "llvm/Support/Casting.h"


namespace clang {
  class Decl;
  class Sema;
}

namespace cling {
  typedef llvm::SmallVector<clang::Stmt*, 2> ASTNodes;

  /// \brief Helper structure that allows children of a node to delegate how
  /// to be replaced from their parent. For example a child can return more than
  /// one replacement node.
  class ASTNodeInfo {
  private:
    ASTNodes Nodes;
    bool forReplacement;
    bool m_HasErrorOccurred;
  public:
    ASTNodeInfo() : forReplacement(false), m_HasErrorOccurred(false) {}
    ASTNodeInfo(clang::Stmt* S, bool needed)
      : forReplacement(needed), m_HasErrorOccurred(false) {
      Nodes.push_back(S);
    }

    bool isForReplacement() { return forReplacement; }
    void setForReplacement(bool val = true) { forReplacement = val; }
    void setErrorOccurred(bool val = true) { m_HasErrorOccurred = val; }
    bool hasErrorOccurred() { return m_HasErrorOccurred; }
    bool hasSingleNode() { return Nodes.size() == 1; }
    clang::Stmt* getAsSingleNode() {
      assert(hasSingleNode() && "There is more than one node!");
      return Nodes[0];
    }
    ASTNodes& getNodes() { return Nodes; }
    void addNode(clang::Stmt* Node) { Nodes.push_back(Node); }
    template <typename T> T* getAs() {
      return llvm::dyn_cast<T>(getAsSingleNode());
    }
    template <typename T> T* castTo() {
      T* Result = llvm::dyn_cast<T>(getAsSingleNode());
      assert(Result && "Cannot cast to type!");
      return Result;
    }
  };
} //end namespace cling

namespace cling {

  typedef llvm::DenseMap<clang::Stmt*, clang::Stmt*> MapTy;

  /// \brief In order to implement the runtime type binding and expression
  /// evaluation we need to be able to compile code which contains unknown
  /// symbols (undefined variables, types, functions, etc.). This cannot be done
  /// by a compiler like clang, because it is not valid C++ code.
  ///
  /// DynamicExprTransformer transforms these unknown symbols into valid C++
  /// code at AST (abstract syntax tree) level. Thus it provides an opportunity
  /// their evaluation to happen at runtime. Several steps are performed:
  ///
  /// 1. Skip compiler's error diagnostics - if a compiler encounters unknown
  /// symbol, by definition, it must output an error and it mustn't produce
  /// machine code. Cling implements an extension to Clang semantic analyzer
  /// that allows the compiler to recover even an unknown symbol is encountered.
  /// For instance if the compiler sees a symbol it looks for its definition in
  /// a internal structure (symbol table) and it is not found it asks whether
  /// somebody else could provide the missing symbol. That is the place where
  /// the DynamicIDHandler, which is controlled by DynamicExprTransformer comes
  /// into play. It marks all unknown symbols as dependent as if they are
  /// templates and are going to be resolved at first instantiation, with the
  /// only difference that an instantiation never happens. The advantage is that
  /// the unknown symbols are not diagnosed but the disadvantage is that
  /// somebody needs to transform them into valid expressions with valid types.
  ///
  /// 2. Replace all dependent symbols - all artificially dependent symbols need
  /// to be replaced with appropriate valid symbols in order the compiler to
  /// produce executable machine code. The DynamicExprTransformer walks up all
  /// statements and declarations that might be possibly marked earlier as
  /// dependent and replaces them with valid expression, which preserves the
  /// meant behavior. Main implementation goal is to replace the as little
  /// as possible part of the statement. The replacement is done immediately
  /// after the expected type can be deduced.
  ///
  /// 2.1. EvaluateT - this is a templated function, which is put at the
  /// place of the dependent expression. It will be called at runtime and it
  /// will use the runtime instance of the interpreter (cling interprets itself)
  /// to evaluate the replaced expression. The template parameter of the
  /// function carries the expected expression type. If unknown symbol is
  /// encountered as a right-hand-side of an assignment one can claim that
  /// the type of the unknown expression should be compatible with the type of
  /// the left-hand-side.
  ///
  /// 2.2 LifetimeHandler - in some more complex situation in order to preserve
  /// the behavior the expression must be replaced with more complex structures.
  ///
  /// 3. Evaluate interface - this is the core function in the interpreter,
  /// which does the delayed evaluation. It uses a callback function, which
  /// should be reimplemented in the subsystem that provides the runtime types
  /// and addresses of the expressions.
  class EvaluateTSynthesizer : public ASTTransformer,
                               public clang::StmtVisitor<EvaluateTSynthesizer,
                                                         ASTNodeInfo> {

  private:

    /// \brief Stores the declaration of the EvaluateT function.
    clang::FunctionDecl* m_EvalDecl;

    /// \brief Stores helper structure dealing with static initializers.
    clang::CXXRecordDecl* m_LifetimeHandlerDecl;

    /// \brief Stores member function defined in the LifetimeHanlder.
    clang::CXXMethodDecl* m_LHgetMemoryDecl;

    /// \brief Stores helper class used in EvaluateT call.
    clang::CXXRecordDecl* m_DynamicExprInfoDecl;

    /// \brief Stores the clang::DeclContext declaration, used in as an parameter
    /// in EvaluateT call.
    clang::CXXRecordDecl* m_DeclContextDecl;

    /// \brief Stores the cling::Interpreter (cling::runtime::gCling),
    /// used in as an parameter LifetimeHandler's ctor.
    clang::VarDecl* m_gCling;

    /// \brief Keeps track of the replacements being made. If an AST node is
    /// changed with another it should be added to the map (newNode->oldNode).
    MapTy m_SubstSymbolMap;

    /// \brief Stores the actual declaration context, in which declarations are
    /// being visited.
    clang::DeclContext* m_CurDeclContext;

    /// \brief Use instead of clang::SourceRange().
    clang::SourceRange m_NoRange;

    /// \brief Use instead of clang::SourceLocation() as start location.
    clang::SourceLocation m_NoSLoc;

    /// \brief Use instead of clang::SourceLocation() as end location.
    clang::SourceLocation m_NoELoc;

    /// \brief Needed for the AST transformations, owned by Sema.
    clang::ASTContext* m_Context;

    /// \brief Counter used when we need unique names.
    unsigned long long m_UniqueNameCounter;

    /// \brief Counter used when entering/exitting compound stmt. Needed for
    /// value printing of dynamic expressions.
    unsigned m_NestedCompoundStmts;

  public:

    typedef clang::StmtVisitor<EvaluateTSynthesizer, ASTNodeInfo>
      BaseStmtVisitor;

    using BaseStmtVisitor::Visit;

    EvaluateTSynthesizer(clang::Sema* S);

    ~EvaluateTSynthesizer();

    Result Transform(clang::Decl* D) override;

    MapTy& getSubstSymbolMap() { return m_SubstSymbolMap; }

    ASTNodeInfo VisitStmt(clang::Stmt* Node);
    ASTNodeInfo VisitCompoundStmt(clang::CompoundStmt* Node);
    ASTNodeInfo VisitIfStmt(clang::IfStmt* Node);

    /// \brief Transforms a declaration with initializer of dependent type.
    /// If an object on the free store is being initialized we use the
    /// EvaluateT
    /// If an object on the stack is being initialized it is transformed into
    /// reference and an object on the free store is created in order to
    /// avoid the copy constructors, which might be private
    ///
    /// For example:
    /// @code
    /// int i = 5;
    /// MyClass my(dep->Symbol(i))
    /// @endcode
    /// where dep->Symbol() is of artificially dependent type it is being
    /// transformed into:
    /// @code
    /// cling::runtime::internal::LifetimeHandler
    /// __unique("dep->Sybmol(*(int*)@)",(void*[]){&i}, DC, "MyClass");
    /// MyClass &my(*(MyClass*)__unique.getMemory());
    /// @endcode
    ///
    /// Note: here our main priority is to preserve equivalent behavior. We have
    /// to clean the heap memory afterwords.
    ///
    ASTNodeInfo VisitDeclStmt(clang::DeclStmt* Node);

    ///\brief \c delete t; \c if t is dependent, the whole stmt must be escaped.
    ///
    ASTNodeInfo VisitCXXDeleteExpr(clang::CXXDeleteExpr* Node);

    ///\brief Surrounds member accesses into dependent types; remove on
    /// subsitution of its child expression.
    ///
    ASTNodeInfo VisitCXXDependentScopeMemberExpr(
                                      clang::CXXDependentScopeMemberExpr* Node);
    ASTNodeInfo VisitExpr(clang::Expr* Node);
    ASTNodeInfo VisitBinaryOperator(clang::BinaryOperator* Node);
    ASTNodeInfo VisitCallExpr(clang::CallExpr* E);
    ASTNodeInfo VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    ASTNodeInfo VisitDependentScopeDeclRefExpr(
                                        clang::DependentScopeDeclRefExpr* Node);

  protected:

    ///\brief On first use it finds commonly used declarations.
    ///
    /// For example: EvaluateT, clang::DeclContext and so on.
    ///
    void Initialize();

    /// @{
    /// @name Helpers, which simplify node replacement

    ///\brief Replaces given dependent AST node with an instantiation of
    /// EvaluateT with the deduced type.
    ///
    /// @param[in] InstTy The deduced type used to create instantiation.
    /// @param[in] SubTree The AST node or subtree, which is being replaced.
    /// @param[in] ValuePrinterReq Whether to turn on the value printing or not
    ///
    clang::Expr* SubstituteUnknownSymbol(const clang::QualType InstTy,
                                         clang::Expr* SubTree,
                                         bool ValuePrinterReq = false);

    ///\brief Builds the actual call expression, which is put in the place of
    /// the dependent AST node.
    ///
    /// @param[in] InstTy The deduced type used to create instantiation.
    /// @param[in] SubTree The AST node or subtree, which is being replaced.
    /// @param[in] CallArgs Proper arguments, which the call will use.
    ///
    clang::CallExpr* BuildEvalCallExpr(clang::QualType InstTy,
                                       clang::Expr* SubTree,
                                  llvm::SmallVector<clang::Expr*, 2>& CallArgs);

    ///\brief Builds the DynamicExprInfo class with proper info.
    ///
    clang::Expr* BuildDynamicExprInfo(clang::Expr* SubTree,
                                      bool ValuePrinterReq = false);

    ///\brief Creates const char* expression from given value.
    clang::Expr* ConstructConstCharPtrExpr(const char* Val);

    ///\brief Checks if the given node is marked as dependent by us.
    ///
    bool IsArtificiallyDependent(clang::Expr* Node);

    ///\brief Checks if the given declaration should be examined. It checks
    /// whether a declaration context marked as dependent contains the
    /// declaration or the declaration type is not one of those we are looking
    /// for.
    ///
    bool ShouldVisit(clang::Decl* D);

    /// \brief Gets all children of a given node.
    ///
    bool GetChildren(ASTNodes& Children, clang::Stmt* Node);

    /// \brief Creates unique name (eg. of a variable). Used internally for
    /// AST node synthesis.
    ///
    void createUniqueName(std::string& out);
    /// @}
  };
} // end namespace cling
#endif // CLING_DYNAMIC_LOOKUP_H
