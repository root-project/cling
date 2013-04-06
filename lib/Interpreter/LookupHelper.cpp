//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: AST.cpp 45014 2012-07-11 20:31:42Z vvassilev $
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/LookupHelper.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"

using namespace clang;

namespace cling {

  ///\brief Cleanup Parser state after a failed lookup.
  /// 
  /// After a failed lookup we need to discard the remaining unparsed input,
  /// restore the original state of the incremental parsing flag, clear any
  /// pending diagnostics, restore the suppress diagnostics flag, and restore
  /// the spell checking language options.
  ///
  class ParserStateRAII {
  private:
    Parser* P;
    Preprocessor& PP;
    bool ResetIncrementalProcessing;
    bool OldSuppressAllDiagnostics;
    bool OldSpellChecking;
    DestroyTemplateIdAnnotationsRAIIObj CleanupTemplateIds;

  public:
    ParserStateRAII(Parser& p)
      : P(&p), PP(p.getPreprocessor()), 
        ResetIncrementalProcessing(p.getPreprocessor()
                                   .isIncrementalProcessingEnabled()),
        OldSuppressAllDiagnostics(p.getPreprocessor().getDiagnostics()
                                  .getSuppressAllDiagnostics()),
        OldSpellChecking(p.getPreprocessor().getLangOpts().SpellChecking),
        CleanupTemplateIds(p)
    {
    }

    ~ParserStateRAII()
    {
      //
      // Advance the parser to the end of the file, and pop the include stack.
      //
      // Note: Consuming the EOF token will pop the include stack.
      //
      P->SkipUntil(tok::eof, /*StopAtSemi*/false, /*DontConsume*/false, 
                   /*StopAtCodeCompletion*/false);
      PP.enableIncrementalProcessing(ResetIncrementalProcessing);
      P->getActions().getDiagnostics().Reset();
      PP.getDiagnostics().setSuppressAllDiagnostics(OldSuppressAllDiagnostics);
      const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking =
         OldSpellChecking;
    }
  };

  // pin *tor here so that we can have clang::Parser defined and be able to call
  // the dtor on the OwningPtr
  LookupHelper::LookupHelper(clang::Parser* P, Interpreter* interp) 
    : m_Parser(P), m_Interpreter(interp) {}

  LookupHelper::~LookupHelper() {}

  QualType LookupHelper::findType(llvm::StringRef typeName) const {
    //
    //  Our return value.
    //
    QualType TheQT;

    // Use P for shortness
    Parser& P = *m_Parser;
    ParserStateRAII ResetParserState(P);
    prepareForParsing(typeName, llvm::StringRef("lookup.type.by.name.file"));
    //
    //  Try parsing the type name.
    //
    TypeResult Res(P.ParseTypeName());
    if (Res.isUsable()) {
      // Accept it only if the whole name was parsed.
      if (P.NextToken().getKind() == clang::tok::eof) {
        TypeSourceInfo* TSI = 0;
        TheQT = clang::Sema::GetTypeFromParser(Res.get(), &TSI);
      }
    }
    return TheQT;
  }

  const Decl* LookupHelper::findScope(llvm::StringRef className,
                                      const Type** resultType /* = 0 */,
                                      bool instantiateTemplate/*=true*/) const {
    //
    //  Some utilities.
    //
    // Use P for shortness
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    Preprocessor& PP = P.getPreprocessor();
    ASTContext& Context = S.getASTContext();
    ParserStateRAII ResetParserState(P);
    prepareForParsing(className.str() + "::", 
                      llvm::StringRef("lookup.class.by.name.file"));
    //
    //  Our return values.
    //
    const Type* TheType = 0;
    const Type** setResultType = &TheType;
    if (resultType)
      setResultType = resultType;
    *setResultType = 0;

    const Decl* TheDecl = 0;

    //
    //  Prevent failing on an assert in TryAnnotateCXXScopeToken.
    //
    if (!P.getCurToken().is(clang::tok::identifier) 
        && !P.getCurToken().is(clang::tok::coloncolon) 
        && !(P.getCurToken().is(clang::tok::annot_template_id) 
             && P.NextToken().is(clang::tok::coloncolon)) 
        && !P.getCurToken().is(clang::tok::kw_decltype)) {
      // error path
      return TheDecl;
    }
    //
    //  Try parsing the name as a nested-name-specifier.
    //
    if (P.TryAnnotateCXXScopeToken(false)) {
      // error path
      return TheDecl;
    }
    if (P.getCurToken().getKind() == tok::annot_cxxscope) {
      CXXScopeSpec SS;
      S.RestoreNestedNameSpecifierAnnotation(P.getCurToken().getAnnotationValue(),
                                             P.getCurToken().getAnnotationRange(),
                                             SS);
      if (SS.isValid()) {
        NestedNameSpecifier* NNS = SS.getScopeRep();
        NestedNameSpecifier::SpecifierKind Kind = NNS->getKind();
        // Only accept the parse if we consumed all of the name.
        if (P.NextToken().getKind() == clang::tok::eof) {
          //
          //  Be careful, not all nested name specifiers refer to classes
          //  and namespaces, and those are the only things we want.
          //
          switch (Kind) {
            case NestedNameSpecifier::Identifier: {
                // Dependent type.
                // We do not accept these.
              }
              break;
            case NestedNameSpecifier::Namespace: {
                // Namespace.
                NamespaceDecl* NSD = NNS->getAsNamespace();
                NSD = NSD->getCanonicalDecl();
                TheDecl = NSD;
              }
              break;
            case NestedNameSpecifier::NamespaceAlias: {
                // Namespace alias.
                // Note: In the future, should we return the alias instead?
                NamespaceAliasDecl* NSAD = NNS->getAsNamespaceAlias();
                NamespaceDecl* NSD = NSAD->getNamespace();
                NSD = NSD->getCanonicalDecl();
                TheDecl = NSD;
              }
              break;
            case NestedNameSpecifier::TypeSpec:
                // Type name.
                // Intentional fall-though
            case NestedNameSpecifier::TypeSpecWithTemplate: {
                // Type name qualified with "template".
                // Note: Do we need to check for a dependent type here?
                NestedNameSpecifier *prefix = NNS->getPrefix();
                if (prefix) {
                   QualType temp 
                     = Context.getElaboratedType(ETK_None,prefix,
                                                 QualType(NNS->getAsType(),0));
                   *setResultType = temp.getTypePtr();
                } else {
                   *setResultType = NNS->getAsType();
                }
                const TagType* TagTy = (*setResultType)->getAs<TagType>();
                if (TagTy) {
                  // It is a class, struct, or union.
                  TagDecl* TD = TagTy->getDecl();
                  if (TD) {
                    TheDecl = TD->getDefinition();
                    if (!TheDecl && instantiateTemplate) {
                      // The user wants to see the template instantiation,
                      // existing or not.

                      // Here we might not have an active transaction to handle
                      // the caused instantiation decl.

                      //FIXME:
                      // We need it, but it massively breaks roottest. Why?!
                      // Interpreter::PushTransactionRAII pushedT(m_Interpreter);

                      // Make sure it is not just forward declared, and
                      // instantiate any templates.
                      if (!S.RequireCompleteDeclContext(SS, TD)) {
                        // Success, type is complete, instantiations have
                        // been done.
                        TheDecl = TD->getDefinition();
                      }
                    }
                  }
                }
              }
              break;
            case clang::NestedNameSpecifier::Global: {
                // Name was just "::" and nothing more.
                TheDecl = Context.getTranslationUnitDecl();
              }
              break;
          }
          return TheDecl;
        }
      }
    }
    //
    //  Cleanup after failed parse as a nested-name-specifier.
    //
    P.SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
                /*StopAtCodeCompletion*/false);
    S.getDiagnostics().Reset();
    //
    //  Setup to reparse as a type.
    //

    llvm::MemoryBuffer* SB =
      llvm::MemoryBuffer::getMemBufferCopy(className.str() + "\n",
                                           "lookup.type.file");
    clang::FileID FID = S.getSourceManager().createFileIDForMemBuffer(SB);
    PP.EnterSourceFile(FID, 0, clang::SourceLocation());
    PP.Lex(const_cast<clang::Token&>(P.getCurToken()));

    //
    //  Now try to parse the name as a type.
    //
    if (P.TryAnnotateTypeOrScopeToken(false, false)) {
      // error path
      return TheDecl;
    }
    if (P.getCurToken().getKind() == tok::annot_typename) {
      ParsedType T = P.getTypeAnnotation(const_cast<Token&>(P.getCurToken()));
      // Only accept the parse if we consumed all of the name.
      if (P.NextToken().getKind() == clang::tok::eof)
        if (!T.get().isNull()) {
          TypeSourceInfo *TSI = 0;
          clang::QualType QT = clang::Sema::GetTypeFromParser(T, &TSI);
          if (const TagType* TT = QT->getAs<TagType>()) {
            TheDecl = TT->getDecl()->getDefinition();
            *setResultType = QT.getTypePtr();
          }
        }
    }
    return TheDecl;
  }

  const ClassTemplateDecl* LookupHelper::findClassTemplate(llvm::StringRef Name) const {
    //
    //  Find a class template decl given its name.
    //

    // Humm ... this seems to do the trick ... or does it? or is there a better way?

    // Use P for shortness
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    ASTContext& Context = S.getASTContext();
    ParserStateRAII ResetParserState(P);
    prepareForParsing(Name.str(), 
                      llvm::StringRef("lookup.class.by.name.file"));

    //
    //  Prevent failing on an assert in TryAnnotateCXXScopeToken.
    //
    if (!P.getCurToken().is(clang::tok::identifier) 
        && !P.getCurToken().is(clang::tok::coloncolon) 
        && !(P.getCurToken().is(clang::tok::annot_template_id) 
             && P.NextToken().is(clang::tok::coloncolon)) 
        && !P.getCurToken().is(clang::tok::kw_decltype)) {
      // error path
      return 0;
    }

    //
    //  Now try to parse the name as a type.
    //
    if (P.TryAnnotateTypeOrScopeToken(false, false)) {
      // error path
      return 0;
    }
    DeclContext *where = 0;
    if (P.getCurToken().getKind() == tok::annot_cxxscope) {
      CXXScopeSpec SS;
      S.RestoreNestedNameSpecifierAnnotation(P.getCurToken().getAnnotationValue(),
                                             P.getCurToken().getAnnotationRange(),
                                             SS);
      if (SS.isValid()) {
        P.ConsumeToken();
        if (!P.getCurToken().is(clang::tok::identifier)) {
          return 0;
        }
        NestedNameSpecifier *nested = SS.getScopeRep();
        if (!nested) return 0;
        switch (nested->getKind()) {
        case NestedNameSpecifier::Global:
          where = Context.getTranslationUnitDecl();
          break;
        case NestedNameSpecifier::Namespace:
          where = nested->getAsNamespace();
          break;
        case NestedNameSpecifier::NamespaceAlias:
        case NestedNameSpecifier::Identifier:
           return 0;
        case NestedNameSpecifier::TypeSpec:
        case NestedNameSpecifier::TypeSpecWithTemplate: 
          {
            const Type *ntype = nested->getAsType();
            where = ntype->getAsCXXRecordDecl();
            if (!where) return 0;
            break;
          }
        };
      }
    } else if (P.getCurToken().is(clang::tok::identifier)) {
      // We have a single indentifier, let's look for it in the
      // the global scope.
      where = Context.getTranslationUnitDecl();
    }
    if (where) {
      // Great we now have a scope and something to search for,let's go ahead.
      for (DeclContext::lookup_result R 
              = where->lookup(P.getCurToken().getIdentifierInfo());
           R.first != R.second; ++R.first) {
        ClassTemplateDecl *theDecl = dyn_cast<ClassTemplateDecl>((*R.first));
        if (theDecl)
          return theDecl;
      }
    }
    return 0;
  }

  const FunctionDecl* LookupHelper::findFunctionProto(const Decl* scopeDecl,
                                                      llvm::StringRef funcName, 
                                               llvm::StringRef funcProto) const {
    assert(scopeDecl && "Decl cannot be null");
    //
    //  Our return value.
    //
    FunctionDecl* TheDecl = 0;
    //
    //  Some utilities.
    //
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    Preprocessor& PP = S.getPreprocessor();
    ASTContext& Context = S.getASTContext();
    ParserStateRAII ResetParserState(P);
    prepareForParsing(funcProto, llvm::StringRef("func.prototype.file"));
    //
    //  Get the DeclContext we will search for the function.
    //
    NestedNameSpecifier* classNNS = 0;
    if (const NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(scopeDecl)) {
      classNNS = NestedNameSpecifier::Create(Context, 0,
                                             const_cast<NamespaceDecl*>(NSD));
    }
    else if (const RecordDecl* RD = dyn_cast<RecordDecl>(scopeDecl)) {
      const Type* T = Context.getRecordType(RD).getTypePtr();
      classNNS = NestedNameSpecifier::Create(Context, 0, false, T);
    }
    else if (llvm::isa<TranslationUnitDecl>(scopeDecl)) {
      classNNS = NestedNameSpecifier::GlobalSpecifier(Context);
    }
    else {
      // Not a namespace or class, we cannot use it.
      return 0;
    }
    DeclContext* foundDC = dyn_cast<DeclContext>(const_cast<Decl*>(scopeDecl));
    //
    //  If we are looking up a member function, construct
    //  the implicit object argument.
    //
    //  Note: For now this is always a non-CV qualified lvalue.
    //
    QualType ClassType;
    Expr* ObjExpr = 0;
    Expr::Classification ObjExprClassification;
    if (CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(foundDC)) {
      ClassType = Context.getTypeDeclType(CRD).getCanonicalType();
      ObjExpr = new (Context) OpaqueValueExpr(SourceLocation(),
        ClassType, VK_LValue);
      ObjExprClassification = ObjExpr->Classify(Context);
      //GivenArgTypes.insert(GivenArgTypes.begin(), ClassType);
      //GivenArgs.insert(GivenArgs.begin(), ObjExpr);
    }
    //
    //  Parse the prototype now.
    //
    llvm::SmallVector<QualType, 4> GivenArgTypes;
    llvm::SmallVector<Expr*, 4> GivenArgs;
    while (P.getCurToken().isNot(tok::eof)) {
      TypeResult Res(P.ParseTypeName());
      if (!Res.isUsable()) {
        // Bad parse, done.
        return TheDecl;
      }
      TypeSourceInfo *TSI = 0;
      clang::QualType QT = clang::Sema::GetTypeFromParser(Res.get(), &TSI);
      QT = QT.getCanonicalType();
      GivenArgTypes.push_back(QT);
      {
        ExprValueKind VK = VK_RValue;
        if (QT->getAs<LValueReferenceType>()) {
          VK = VK_LValue;
        }
        clang::QualType NonRefQT(QT.getNonReferenceType());
        Expr* val = new (Context) OpaqueValueExpr(SourceLocation(), NonRefQT,
          VK);
        GivenArgs.push_back(val);
      }
      // Type names should be comma separated.
      // FIXME: Here if we have type followed by name won't work. Eg int f, ...
      if (!P.getCurToken().is(clang::tok::comma)) {
        break;
      }
      // Eat the comma.
      P.ConsumeToken();
    }
    if (P.getCurToken().isNot(tok::eof)) {
      // We did not consume all of the prototype, bad parse.
      return TheDecl;
    }
    //
    //  Cleanup after prototype parse.
    //
    P.SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false, 
                /*StopAtCodeCompletion*/false);
    S.getDiagnostics().Reset();
    //
    //  Setup to reparse as a type.
    //
    //
    //  Create a fake file to parse the function name.
    //
    {
      llvm::MemoryBuffer* SB 
        = llvm::MemoryBuffer::getMemBufferCopy(funcName.str()
                                               + "\n", "lookup.funcname.file");
      clang::FileID FID = S.getSourceManager().createFileIDForMemBuffer(SB);
      PP.EnterSourceFile(FID, /*DirLookup=*/0, clang::SourceLocation());
      PP.Lex(const_cast<clang::Token&>(P.getCurToken()));
    }
    {
      //
      //  Parse the function name.
      //
      SourceLocation TemplateKWLoc;
      UnqualifiedId FuncId;
      CXXScopeSpec SS;
      SS.MakeTrivial(Context, classNNS, SourceRange());
      //
      //  Make the class we are looking up the function
      //  in the current scope to please the constructor
      //  name lookup.  We do not need to do this otherwise,
      //  and may be able to remove it in the future if
      //  the way constructors are looked up changes.
      //
      P.EnterScope(Scope::DeclScope);
      S.EnterDeclaratorContext(P.getCurScope(), foundDC);
      if (P.ParseUnqualifiedId(SS, /*EnteringContext*/false,
                               /*AllowDestructorName*/true,
                               /*AllowConstructorName*/true,
                               clang::ParsedType(), TemplateKWLoc,
                               FuncId)) {
        // Bad parse.
        // Destroy the scope we created first, and
        // restore the original.
        S.ExitDeclaratorContext(P.getCurScope());
        P.ExitScope();

        // Then cleanup and exit.
        return TheDecl;
      }
      //
      //  Get any template args in the function name.
      //
      TemplateArgumentListInfo FuncTemplateArgsBuffer;
      DeclarationNameInfo FuncNameInfo;
      const TemplateArgumentListInfo* FuncTemplateArgs;
      S.DecomposeUnqualifiedId(FuncId, FuncTemplateArgsBuffer, FuncNameInfo, 
                               FuncTemplateArgs);
      //
      //  Lookup the function name in the given class now.
      //
      DeclarationName FuncName = FuncNameInfo.getName();
      SourceLocation FuncNameLoc = FuncNameInfo.getLoc();
      LookupResult Result(S, FuncName, FuncNameLoc, Sema::LookupMemberName, 
                          Sema::NotForRedeclaration);
      if (!S.LookupQualifiedName(Result, foundDC)) {
        // Lookup failed.
        // Destroy the scope we created first, and
        // restore the original.
        S.ExitDeclaratorContext(P.getCurScope());
        P.ExitScope();
        // Then cleanup and exit.
        return TheDecl;
      }
      // Destroy the scope we created, and
      // restore the original.
      S.ExitDeclaratorContext(P.getCurScope());
      P.ExitScope();
      //
      //  Check for lookup failure.
      //
      if (Result.getResultKind() != LookupResult::Found &&
          Result.getResultKind() != LookupResult::FoundOverloaded) {
        // Lookup failed.
        return TheDecl;
      }
      {
        //
        //  Construct the overload candidate set.
        //
        OverloadCandidateSet Candidates(FuncNameInfo.getLoc());
        for (LookupResult::iterator I = Result.begin(), E = Result.end();
            I != E; ++I) {
          NamedDecl* ND = *I;
          if (FunctionDecl* FD = dyn_cast<FunctionDecl>(ND)) {
            if (isa<CXXMethodDecl>(FD) &&
                !cast<CXXMethodDecl>(FD)->isStatic() &&
                !isa<CXXConstructorDecl>(FD)) {
              // Class method, not static, not a constructor, so has
              // an implicit object argument.
              CXXMethodDecl* MD = cast<CXXMethodDecl>(FD);
              if (FuncTemplateArgs && (FuncTemplateArgs->size() != 0)) {
                // Explicit template args were given, cannot use a plain func.
                continue;
              }
              S.AddMethodCandidate(MD, I.getPair(), MD->getParent(),
                                   /*ObjectType=*/ClassType,
                                  /*ObjectClassification=*/ObjExprClassification,
                   llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                                   Candidates);
            }
            else {
              const FunctionProtoType* Proto = dyn_cast<FunctionProtoType>(
                FD->getType()->getAs<clang::FunctionType>());
              if (!Proto) {
                // Function has no prototype, cannot do overloading.
                continue;
              }
              if (FuncTemplateArgs && (FuncTemplateArgs->size() != 0)) {
                // Explicit template args were given, cannot use a plain func.
                continue;
              }
              S.AddOverloadCandidate(FD, I.getPair(),
                   llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                                     Candidates);
            }
          }
          else if (FunctionTemplateDecl* FTD =
              dyn_cast<FunctionTemplateDecl>(ND)) {
            if (isa<CXXMethodDecl>(FTD->getTemplatedDecl()) &&
                !cast<CXXMethodDecl>(FTD->getTemplatedDecl())->isStatic() &&
                !isa<CXXConstructorDecl>(FTD->getTemplatedDecl())) {
              // Class method template, not static, not a constructor, so has
              // an implicit object argument.
              S.AddMethodTemplateCandidate(FTD, I.getPair(),
                                      cast<CXXRecordDecl>(FTD->getDeclContext()),
                         const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
                                           /*ObjectType=*/ClassType,
                                  /*ObjectClassification=*/ObjExprClassification,
                   llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                                           Candidates);
            }
            else {
              S.AddTemplateOverloadCandidate(FTD, I.getPair(),
                const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
                llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                Candidates, /*SuppressUserConversions=*/false);
            }
          }
        }
        //
        //  Find the best viable function from the set.
        //
        {
          OverloadCandidateSet::iterator Best;
          OverloadingResult OR = Candidates.BestViableFunction(S, 
                                                             Result.getNameLoc(),
                                                               Best);
          if (OR == OR_Success) {
            TheDecl = Best->Function;
            // We prefer to get the canonical decl for consistency and ease
            // of comparison.
            TheDecl = TheDecl->getCanonicalDecl();
          }
        }
      }
    }
    return TheDecl;
  }

  const FunctionDecl* LookupHelper::findFunctionArgs(const Decl* scopeDecl,
                                                       llvm::StringRef funcName,
                                                llvm::StringRef funcArgs) const {
    //
    //  Our return value.
    //
    FunctionDecl* TheDecl = 0;
    //
    //  Some utilities.
    //
    // Use P for shortness
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    Preprocessor& PP = S.getPreprocessor();
    ASTContext& Context = S.getASTContext();

    ParserStateRAII ResetParserState(P);
    prepareForParsing(funcArgs, llvm::StringRef("func.args.file"));
    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
    //
    NestedNameSpecifier* classNNS = 0;
    if (const NamespaceDecl* NSD = dyn_cast<const NamespaceDecl>(scopeDecl)) {
      classNNS = NestedNameSpecifier::Create(Context, 0,
                                             const_cast<NamespaceDecl*>(NSD));
    }
    else if (const RecordDecl* RD = dyn_cast<const RecordDecl>(scopeDecl)) {
      const Type* T = Context.getRecordType(RD).getTypePtr();
      classNNS = NestedNameSpecifier::Create(Context, 0, false, T);
    }
    else if (llvm::isa<TranslationUnitDecl>(scopeDecl)) {
      classNNS = NestedNameSpecifier::GlobalSpecifier(Context);
    }
    else {
      // Not a namespace or class, we cannot use it.
      return 0;
    }
    CXXScopeSpec SS;
    SS.MakeTrivial(Context, classNNS, SourceRange());
    DeclContext* foundDC = dyn_cast<DeclContext>(const_cast<Decl*>(scopeDecl));
    //
    //  Some validity checks on the passed decl.
    //
    if (foundDC->isDependentContext()) {
      // Passed decl is a template, we cannot use it.
      return 0;
    }
    if (S.RequireCompleteDeclContext(SS, foundDC)) {
      // Forward decl or instantiation failure, we cannot use it.
      return 0;
    }
    //
    //  Get ready for arg list parsing.
    //
    llvm::SmallVector<QualType, 4> GivenArgTypes;
    llvm::SmallVector<Expr*, 4> GivenArgs;
    //
    //  If we are looking up a member function, construct
    //  the implicit object argument.
    //
    //  Note: For now this is always a non-CV qualified lvalue.
    //
    QualType ClassType;
    Expr* ObjExpr = 0;
    Expr::Classification ObjExprClassification = Expr::Classification();
    if (CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(foundDC)) {
      ClassType = Context.getTypeDeclType(CRD).getCanonicalType();
      ObjExpr = new (Context) OpaqueValueExpr(SourceLocation(),
        ClassType, VK_LValue);
      ObjExprClassification = ObjExpr->Classify(Context);
      //GivenArgTypes.insert(GivenArgTypes.begin(), ClassType);
      //GivenArgs.insert(GivenArgs.begin(), ObjExpr);
    }

    //
    //  Parse the arguments now.
    //
    {
      PrintingPolicy Policy(Context.getPrintingPolicy());
      Policy.SuppressTagKeyword = true;
      Policy.SuppressUnwrittenScope = true;
      Policy.SuppressInitializers = true;
      Policy.AnonymousTagLocations = false;
      std::string proto;
      {
        bool first_time = true;
        while (P.getCurToken().isNot(tok::eof)) {
          ExprResult Res = P.ParseAssignmentExpression();
          if (Res.isUsable()) {
            Expr* expr = Res.release();
            GivenArgs.push_back(expr);
            QualType QT = expr->getType().getCanonicalType();
            QualType NonRefQT(QT.getNonReferenceType());
            GivenArgTypes.push_back(NonRefQT);
            if (first_time) {
              first_time = false;
            }
            else {
              proto += ',';
            }
            std::string empty;
            llvm::raw_string_ostream tmp(empty);
            expr->printPretty(tmp, /*PrinterHelper=*/0, Policy, 
                              /*Indentation=*/0);
            proto += tmp.str();
          }
          if (!P.getCurToken().is(tok::comma)) {
            break;
          }
          P.ConsumeToken();
        }
      }
    }
    // For backward compatibility with CINT accept (for now?) a trailing close 
    // parenthesis.
    if (P.getCurToken().isNot(tok::eof) && P.getCurToken().isNot(tok::r_paren) ) {
      // We did not consume all of the arg list, bad parse.
      return TheDecl;
    }
    {
      //
      //  Cleanup after the arg list parse.
      //
      P.SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
                   /*StopAtCodeCompletion*/false);
      S.getDiagnostics().Reset();
      //
      //  Setup to reparse as a type.
      //
      //
      //  Create a fake file to parse the function name.
      //
      {
        llvm::MemoryBuffer* SB 
          = llvm::MemoryBuffer::getMemBufferCopy(funcName.str()
                                                 + "\n", "lookup.funcname.file");
        clang::FileID FID = S.getSourceManager().createFileIDForMemBuffer(SB);
        PP.EnterSourceFile(FID, /*DirLookup=*/0, clang::SourceLocation());
        PP.Lex(const_cast<clang::Token&>(P.getCurToken()));
      }
      //
      //  Make the class we are looking up the function
      //  in the current scope to please the constructor
      //  name lookup.  We do not need to do this otherwise,
      //  and may be able to remove it in the future if
      //  the way constructors are looked up changes.
      //
      P.EnterScope(Scope::DeclScope);
      S.EnterDeclaratorContext(P.getCurScope(), foundDC);
      //
      //  Parse the function name.
      //
      SourceLocation TemplateKWLoc;
      UnqualifiedId FuncId;
      if (P.ParseUnqualifiedId(SS, /*EnteringContext*/false,
                               /*AllowDestructorName*/true,
                               /*AllowConstructorName*/true,
                               ParsedType(), TemplateKWLoc, FuncId)){
        // Failed parse, cleanup.
        S.ExitDeclaratorContext(P.getCurScope());
        P.ExitScope();
        return TheDecl;
      }
      //
      //  Get any template args in the function name.
      //
      TemplateArgumentListInfo FuncTemplateArgsBuffer;
      DeclarationNameInfo FuncNameInfo;
      const TemplateArgumentListInfo* FuncTemplateArgs;
      S.DecomposeUnqualifiedId(FuncId, FuncTemplateArgsBuffer, FuncNameInfo, 
                               FuncTemplateArgs);
      //
      //  Lookup the function name in the given class now.
      //
      DeclarationName FuncName = FuncNameInfo.getName();
      SourceLocation FuncNameLoc = FuncNameInfo.getLoc();
      LookupResult Result(S, FuncName, FuncNameLoc, Sema::LookupMemberName, 
                          Sema::NotForRedeclaration);
      if (!S.LookupQualifiedName(Result, foundDC)) {
        // Lookup failed.
        // Destroy the scope we created first, and
        // restore the original.
        S.ExitDeclaratorContext(P.getCurScope());
        P.ExitScope();
        // Then cleanup and exit.
        return TheDecl;
      }
      //
      //  Destroy the scope we created, and restore the original.
      //
      S.ExitDeclaratorContext(P.getCurScope());
      P.ExitScope();
      //
      //  Check for lookup failure.
      //
      if (Result.getResultKind() != LookupResult::Found &&
          Result.getResultKind() != LookupResult::FoundOverloaded) {
        // Lookup failed.
        return TheDecl;
      }
      {
        //
        //  Construct the overload candidate set.
        //
        OverloadCandidateSet Candidates(FuncNameInfo.getLoc());
        for (LookupResult::iterator I = Result.begin(), E = Result.end();
            I != E; ++I) {
          NamedDecl* ND = *I;
          if (FunctionDecl* FD = dyn_cast<FunctionDecl>(ND)) {
            if (isa<CXXMethodDecl>(FD) &&
                !cast<CXXMethodDecl>(FD)->isStatic() &&
                !isa<CXXConstructorDecl>(FD)) {
              // Class method, not static, not a constructor, so has
              // an implicit object argument.
              CXXMethodDecl* MD = cast<CXXMethodDecl>(FD);
              if (FuncTemplateArgs && (FuncTemplateArgs->size() != 0)) {
                // Explicit template args were given, cannot use a plain func.
                continue;
              }
              S.AddMethodCandidate(MD, I.getPair(), MD->getParent(),
                                   /*ObjectType=*/ClassType,
                                  /*ObjectClassification=*/ObjExprClassification,
                   llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                                   Candidates);
            }
            else {
              const FunctionProtoType* Proto = dyn_cast<FunctionProtoType>(
                FD->getType()->getAs<clang::FunctionType>());
              if (!Proto) {
                // Function has no prototype, cannot do overloading.
                continue;
              }
              if (FuncTemplateArgs && (FuncTemplateArgs->size() != 0)) {
                // Explicit template args were given, cannot use a plain func.
                continue;
              }
              S.AddOverloadCandidate(FD, I.getPair(),
                   llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                                     Candidates);
            }
          }
          else if (FunctionTemplateDecl* FTD =
              dyn_cast<FunctionTemplateDecl>(ND)) {
            if (isa<CXXMethodDecl>(FTD->getTemplatedDecl()) &&
                !cast<CXXMethodDecl>(FTD->getTemplatedDecl())->isStatic() &&
                !isa<CXXConstructorDecl>(FTD->getTemplatedDecl())) {
              // Class method template, not static, not a constructor, so has
              // an implicit object argument.
              S.AddMethodTemplateCandidate(FTD, I.getPair(),
                                      cast<CXXRecordDecl>(FTD->getDeclContext()),
                         const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
                                           /*ObjectType=*/ClassType,
                                  /*ObjectClassification=*/ObjExprClassification,
                   llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                                           Candidates);
            }
            else {
              S.AddTemplateOverloadCandidate(FTD, I.getPair(),
                const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
                llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                Candidates, /*SuppressUserConversions=*/false);
            }
          }
          else {
          }
        }
        //
        //  Find the best viable function from the set.
        //
        {
          OverloadCandidateSet::iterator Best;
          OverloadingResult OR = Candidates.BestViableFunction(S, 
                                                             Result.getNameLoc(),
                                                               Best);
          if (OR == OR_Success) {
            TheDecl = Best->Function;
            // We prefer to get the canonical decl for consistency and ease
            // of comparison.
            TheDecl = TheDecl->getCanonicalDecl();
          }
        }
      }
    }
    return TheDecl;
  }

  void LookupHelper::findArgList(llvm::StringRef argList,
                                 llvm::SmallVector<Expr*, 4>& argExprs) const {
    //
    //  Some utilities.
    //
    // Use P for shortness
    Parser& P = *m_Parser;
    ParserStateRAII ResetParserState(P);
    prepareForParsing(argList, llvm::StringRef("arg.list.file"));
    //
    //  Parse the arguments now.
    //
    {
      bool hasUnusableResult = false;
      while (P.getCurToken().isNot(tok::eof)) {
        ExprResult Res = P.ParseAssignmentExpression();
        if (Res.isUsable()) {
          argExprs.push_back(Res.release());
        }
        else {
          hasUnusableResult = true;
          break;
        }
        if (!P.getCurToken().is(tok::comma)) {
          break;
        }
        P.ConsumeToken();
      }
      if (hasUnusableResult)
        // if one of the arguments is not usable return empty.
        argExprs.clear();
    }
  }

  void LookupHelper::prepareForParsing(llvm::StringRef code, 
                                       llvm::StringRef bufferName) const {
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    Preprocessor& PP = P.getPreprocessor();
    //
    //  Tell the diagnostic engine to ignore all diagnostics.
    //
    PP.getDiagnostics().setSuppressAllDiagnostics(true);
    //
    //  Tell the parser to not attempt spelling correction.
    //
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = 0;
    //
    //  Create a fake file to parse the type name.
    //
    llvm::MemoryBuffer* SB 
      = llvm::MemoryBuffer::getMemBufferCopy(code.str() + "\n",
                                             bufferName.str());
    FileID FID = S.getSourceManager().createFileIDForMemBuffer(SB);
    //
    //  Turn on ignoring of the main file eof token.
    //
    //  Note: We need this because token readahead in the following
    //        routine calls ends up parsing it multiple times.
    //
    if (!PP.isIncrementalProcessingEnabled()) {
      PP.enableIncrementalProcessing();
    }
    //
    //  Switch to the new file the way #include does.
    //
    //  Note: To switch back to the main file we must consume an eof token.
    //
    PP.EnterSourceFile(FID, /*DirLookup=*/0, SourceLocation());
    PP.Lex(const_cast<Token&>(P.getCurToken()));

  }


  bool LookupHelper::hasFunction(const clang::Decl* scopeDecl,
                                 llvm::StringRef funcName) const {

    //FIXME: remore code duplication with findFunctionArgs() and friends.

    assert(scopeDecl && "Decl cannot be null");
    //
    //  Our return value.
    //
    FunctionDecl* TheDecl = 0;
    //
    //  Some utilities.
    //
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    Preprocessor& PP = S.getPreprocessor();
    ASTContext& Context = S.getASTContext();
    ParserStateRAII ResetParserState(P);
    //
    //  Get the DeclContext we will search for the function.
    //
    NestedNameSpecifier* classNNS = 0;
    if (const NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(scopeDecl)) {
      classNNS = NestedNameSpecifier::Create(Context, 0,
                                             const_cast<NamespaceDecl*>(NSD));
    }
    else if (const RecordDecl* RD = dyn_cast<RecordDecl>(scopeDecl)) {
      const Type* T = Context.getRecordType(RD).getTypePtr();
      classNNS = NestedNameSpecifier::Create(Context, 0, false, T);
    }
    else if (llvm::isa<TranslationUnitDecl>(scopeDecl)) {
      classNNS = NestedNameSpecifier::GlobalSpecifier(Context);
    }
    else {
      // Not a namespace or class, we cannot use it.
      return 0;
    }
    DeclContext* foundDC = dyn_cast<DeclContext>(const_cast<Decl*>(scopeDecl));
    //
    //  If we are looking up a member function, construct
    //  the implicit object argument.
    //
    //  Note: For now this is always a non-CV qualified lvalue.
    //
    QualType ClassType;
    Expr* ObjExpr = 0;
    Expr::Classification ObjExprClassification;
    if (CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(foundDC)) {
      ClassType = Context.getTypeDeclType(CRD).getCanonicalType();
      ObjExpr = new (Context) OpaqueValueExpr(SourceLocation(),
        ClassType, VK_LValue);
      ObjExprClassification = ObjExpr->Classify(Context);
      //GivenArgTypes.insert(GivenArgTypes.begin(), ClassType);
      //GivenArgs.insert(GivenArgs.begin(), ObjExpr);
    }
    //
    //  Setup to reparse as a type.
    //
    //
    //  Create a fake file to parse the function name.
    //
    {
      llvm::MemoryBuffer* SB 
        = llvm::MemoryBuffer::getMemBufferCopy(funcName.str()
                                               + "\n", "lookup.funcname.file");
      clang::FileID FID = S.getSourceManager().createFileIDForMemBuffer(SB);
      PP.EnterSourceFile(FID, /*DirLookup=*/0, clang::SourceLocation());
      PP.Lex(const_cast<clang::Token&>(P.getCurToken()));
    }
    {
      //
      //  Parse the function name.
      //
      SourceLocation TemplateKWLoc;
      UnqualifiedId FuncId;
      CXXScopeSpec SS;
      SS.MakeTrivial(Context, classNNS, SourceRange());
      //
      //  Make the class we are looking up the function
      //  in the current scope to please the constructor
      //  name lookup.  We do not need to do this otherwise,
      //  and may be able to remove it in the future if
      //  the way constructors are looked up changes.
      //
      P.EnterScope(Scope::DeclScope);
      S.EnterDeclaratorContext(P.getCurScope(), foundDC);
      if (P.ParseUnqualifiedId(SS, /*EnteringContext*/false,
                               /*AllowDestructorName*/true,
                               /*AllowConstructorName*/true,
                               clang::ParsedType(), TemplateKWLoc,
                               FuncId)) {
        // Bad parse.
        // Destroy the scope we created first, and
        // restore the original.
        S.ExitDeclaratorContext(P.getCurScope());
        P.ExitScope();

        // Then cleanup and exit.
        return TheDecl;
      }
      //
      //  Get any template args in the function name.
      //
      TemplateArgumentListInfo FuncTemplateArgsBuffer;
      DeclarationNameInfo FuncNameInfo;
      const TemplateArgumentListInfo* FuncTemplateArgs;
      S.DecomposeUnqualifiedId(FuncId, FuncTemplateArgsBuffer, FuncNameInfo, 
                               FuncTemplateArgs);
      //
      //  Lookup the function name in the given class now.
      //
      DeclarationName FuncName = FuncNameInfo.getName();
      SourceLocation FuncNameLoc = FuncNameInfo.getLoc();
      LookupResult Result(S, FuncName, FuncNameLoc, Sema::LookupMemberName, 
                          Sema::NotForRedeclaration);
      if (!S.LookupQualifiedName(Result, foundDC)) {
        // Lookup failed.
        // Destroy the scope we created first, and
        // restore the original.
        S.ExitDeclaratorContext(P.getCurScope());
        P.ExitScope();
        // Then cleanup and exit.
        return TheDecl;
      }
      // Destroy the scope we created, and
      // restore the original.
      S.ExitDeclaratorContext(P.getCurScope());
      P.ExitScope();
      //
      //  Check for lookup failure.
      //
      if (Result.empty())
        return false;
      if (Result.isSingleResult())
        return isa<FunctionDecl>(Result.getFoundDecl());
      // We have many - those must be functions.
      return true;
    }
    return false;
  }


} // end namespace cling
