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

    if (typeName.empty()) return TheQT;

    // Could trigger deserialization of decls.
    Interpreter::PushTransactionRAII RAII(m_Interpreter);

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

    // The user wants to see the template instantiation, existing or not.
    // Here we might not have an active transaction to handle
    // the caused instantiation decl.
    Interpreter::PushTransactionRAII pushedT(m_Interpreter);

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

    if (Name.empty()) return 0;

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
      DeclContext::lookup_result R
        = where->lookup(P.getCurToken().getIdentifierInfo());
      for (DeclContext::lookup_iterator I = R.begin(), E = R.end();
           I != E; ++I) {
        ClassTemplateDecl *theDecl = dyn_cast<ClassTemplateDecl>(*I);
        if (theDecl)
          return theDecl;
      }
    }
    return 0;
  }

  static
  DeclContext* getContextAndSpec(CXXScopeSpec &SS,
                                 const Decl* scopeDecl,
                                 ASTContext& Context, Sema &S) {
    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
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
    //  Some validity checks on the passed decl.
    //
    if (foundDC->isDependentContext()) {
      // Passed decl is a template, we cannot use it.
      return 0;
    }
    SS.MakeTrivial(Context, classNNS, SourceRange());
    if (S.RequireCompleteDeclContext(SS, foundDC)) {
      // Forward decl or instantiation failure, we cannot use it.
      return 0;
    }

    return foundDC;
  }

  static bool FuncArgTypesMatch(const ASTContext& C, 
                             const llvm::SmallVector<QualType, 4>& GivenArgTypes,
                                const FunctionProtoType* FPT) {
    // FIXME: What if FTP->arg_size() != GivenArgTypes.size()?
    FunctionProtoType::arg_type_iterator ATI = FPT->arg_type_begin();
    FunctionProtoType::arg_type_iterator E = FPT->arg_type_end();
    llvm::SmallVector<QualType, 4>::const_iterator GAI = GivenArgTypes.begin();
    for (; ATI && (ATI != E); ++ATI, ++GAI) {
      if (!C.hasSameType(*ATI, *GAI)) {
        return false;
      }
    }
    return true;
  }

  static bool IsOverload(const ASTContext& C,
                         const TemplateArgumentListInfo* FuncTemplateArgs,
                         const llvm::SmallVector<QualType, 4>& GivenArgTypes, 
                         const FunctionDecl* FD) {

    //FunctionTemplateDecl* FTD = FD->getDescribedFunctionTemplate();
    QualType FQT = C.getCanonicalType(FD->getType());
    if (llvm::isa<FunctionNoProtoType>(FQT.getTypePtr())) {
      // A K&R-style function (no prototype), is considered to match the args.
      return false;
    }
    const FunctionProtoType* FPT = llvm::cast<FunctionProtoType>(FQT);
    if ((GivenArgTypes.size() != FPT->getNumArgs()) ||
        //(GivenArgsAreEllipsis != FPT->isVariadic()) ||
        !FuncArgTypesMatch(C, GivenArgTypes, FPT)) {
      return true;
    }
    return false;
  }

  static
  const FunctionDecl* overloadFunctionSelector(DeclContext* foundDC,
                                               bool objectIsConst,
                                  const llvm::SmallVector<Expr*, 4> &GivenArgs,
                                     LookupResult &Result,
                                     DeclarationNameInfo &FuncNameInfo,
                              const TemplateArgumentListInfo* FuncTemplateArgs,
                                     ASTContext& Context, Parser &P, Sema &S) {
    //
    //  Our return value.
    //
    FunctionDecl* TheDecl = 0;

    //
    //  If we are looking up a member function, construct
    //  the implicit object argument.
    //
    //  Note: For now this is always a non-CV qualified lvalue.
    //
    QualType ClassType;
    Expr::Classification ObjExprClassification;
    if (CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(foundDC)) {
      if (objectIsConst) 
        ClassType = Context.getTypeDeclType(CRD).getCanonicalType().withConst();
      else ClassType = Context.getTypeDeclType(CRD).getCanonicalType();
      OpaqueValueExpr ObjExpr(SourceLocation(),
                              ClassType, VK_LValue);
      ObjExprClassification = ObjExpr.Classify(Context);
    }

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
        // Is there any other cases?
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
    return TheDecl;
  }

  static
  const FunctionDecl* matchFunctionSelector(DeclContext* foundDC,
                                            bool objectIsConst,
                                  const llvm::SmallVector<Expr*, 4> &GivenArgs,
                                     LookupResult &Result,
                                     DeclarationNameInfo &FuncNameInfo,
                              const TemplateArgumentListInfo* FuncTemplateArgs,
                                     ASTContext& Context, Parser &P, Sema &S) {
    //
    //  Our return value.
    //
    const FunctionDecl* TheDecl = overloadFunctionSelector(foundDC, objectIsConst,
                                                           GivenArgs, Result,
                                                           FuncNameInfo,
                                                           FuncTemplateArgs,
                                                           Context,P,S);
    
    if (TheDecl) {
      llvm::SmallVector<QualType, 4> GivenArgTypes;
      for( size_t s = 0 ; s < GivenArgs.size(); ++s) {
        GivenArgTypes.push_back( GivenArgs[s]->getType().getCanonicalType() );
      }
      if ( IsOverload( Context, FuncTemplateArgs, GivenArgTypes, TheDecl) ) {
        return 0;
      } else {
        // Double check const-ness.
        if (const clang::CXXMethodDecl *md =
            llvm::dyn_cast<clang::CXXMethodDecl>(TheDecl)) {
          if (md->getTypeQualifiers() & clang::Qualifiers::Const) {
            if (!objectIsConst) {
              TheDecl = 0;
            }
          } else {
            if (objectIsConst) {
              TheDecl = 0;
            }
          }
        }
      }
    }
    return TheDecl;
  }

  static bool ParseWithShortcuts(DeclContext* foundDC, CXXScopeSpec &SS,
                          llvm::StringRef funcName,
                          Parser &P, Sema &S,
                          UnqualifiedId &FuncId) {
     
    // Use a very simple parse step that dectect whether the name search (which
    // is already supposed to be an unqualified name) is a simple identifier,
    // a constructor name or a destructor name.  In those 3 cases, we can easily
    // create the UnqualifiedId object that would have resulted from the 'real'
    // parse.  By using this direct creation of the UnqualifiedId, we avoid the
    // 'permanent' cost associated with creating a memory buffer and the
    // associated FileID.
     
    // If the name is a template or an operator, we revert to the regular parse
    // (and its associated permanent cost).
     
    // In the operator case, the additional work is in the case of a conversion
    // operator where we would need to 'quickly' parse the type itself (if want
    // to avoid the permanent cost).
     
    // In the case with the template the problem gets a bit worse as we need to
    // handle potentially arbitrary spaces and ordering
    // ('const int' vs 'int  const', etc.)
     
    if (funcName.size() == 0) return false;
    Preprocessor& PP = S.getPreprocessor();

    // See if we can avoid creating the buffer, for now we just look for
    // simple indentifier, constructor and destructor.
     
     
    if (funcName.size() > 8 && strncmp(funcName.data(),"operator",8) == 0
               &&(   funcName[8] == ' ' || funcName[8] == '*'
                  || funcName[8] == '%' || funcName[8] == '&'
                  || funcName[8] == '+' || funcName[8] == '-'
                  || funcName[8] == '(' || funcName[8] == '['
                  || funcName[8] == '=' || funcName[8] == '!'
                  || funcName[8] == '<' || funcName[8] == '>'
                  || funcName[8] == '-' || funcName[8] == '^')
               ) {
      // We have called:
      //   setOperatorFunctionId (SourceLocation OperatorLoc,
      //                          OverloadedOperatorKind Op,
      //                          SourceLocation SymbolLocations[3])
      // or
      //   setConversionFunctionId (SourceLocation OperatorLoc,
      //                            ParsedType Ty, SourceLocation EndLoc)
    } else if (funcName.find('<') != StringRef::npos) {
      // We might have a template name,
      //   setTemplateId (TemplateIdAnnotation *TemplateId)
      // or
      //   setConstructorTemplateId (TemplateIdAnnotation *TemplateId)
    } else if (funcName[0] == '~') {
       // Destructor.
       // Let's see if this is our contructor.
       TagDecl *decl = llvm::dyn_cast<TagDecl>(foundDC);
       if (decl) {
          // We have a class or struct or something.
          if (funcName.substr(1).equals(decl->getName())) {
             ParsedType PT;
             QualType QT( decl->getTypeForDecl(), 0 );
             PT.set(QT);
             FuncId.setDestructorName(SourceLocation(),PT,SourceLocation());
             return true;
          }
       }
       // So it starts with ~ but is not followed by the name of
       // a class or at least not the one that is the declaration context,
       // let's try a real parsing, to see if we can do better.
    } else {
       // We either have a simple type or a constructor name
       TagDecl *decl = llvm::dyn_cast<TagDecl>(foundDC);
       if (decl) {
          // We have a class or struct or something.
          if (funcName.equals(decl->getName())) {
             ParsedType PT;
             QualType QT( decl->getTypeForDecl(), 0 );
             PT.set(QT);
             FuncId.setConstructorName(PT,SourceLocation(),SourceLocation());
          } else {
             IdentifierInfo *TypeInfoII = &PP.getIdentifierTable().get(funcName);
             FuncId.setIdentifier (TypeInfoII, SourceLocation() );
          }
          return true;
       } else {
          // We have a namespace like context, it can't be a constructor
          IdentifierInfo *TypeInfoII = &PP.getIdentifierTable().get(funcName);
          FuncId.setIdentifier (TypeInfoII, SourceLocation() );
          return true;
       }
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
      

    //
    //  Parse the function name.
    //
    SourceLocation TemplateKWLoc;
    if (P.ParseUnqualifiedId(SS, /*EnteringContext*/false,
                             /*AllowDestructorName*/true,
                             /*AllowConstructorName*/true,
                             ParsedType(), TemplateKWLoc,
                             FuncId)) {
      // Failed parse, cleanup.
      return false;
    }
    return true;
  }

   
  template <typename T>
  T findFunction(DeclContext* foundDC, CXXScopeSpec &SS,
                 llvm::StringRef funcName,
                 const llvm::SmallVector<Expr*, 4> &GivenArgs,
                 bool objectIsConst,
                 ASTContext& Context, Parser &P, Sema &S,
                 T (*functionSelector)(DeclContext* foundDC,
                                       bool objectIsConst,
                                  const llvm::SmallVector<Expr*, 4> &GivenArgs,
                                       LookupResult &Result,
                                       DeclarationNameInfo &FuncNameInfo,
                              const TemplateArgumentListInfo* FuncTemplateArgs,
                                       ASTContext& Context, Parser &P, Sema &S)
                  ) {
    // Given the correctly types arguments, etc. find the function itself.

    //
    //  Our return value.
    //
    FunctionDecl* TheDecl = 0;

    //
    //  Make the class we are looking up the function
    //  in the current scope to please the constructor
    //  name lookup.  We do not need to do this otherwise,
    //  and may be able to remove it in the future if
    //  the way constructors are looked up changes.
    //
    void* OldEntity = P.getCurScope()->getEntity();
    DeclContext* TUCtx = Context.getTranslationUnitDecl();
    P.getCurScope()->setEntity(TUCtx);
    P.EnterScope(Scope::DeclScope);
    P.getCurScope()->setEntity(foundDC);
    P.EnterScope(Scope::DeclScope);
    Sema::ContextRAII SemaContext(S, foundDC);
    S.EnterDeclaratorContext(P.getCurScope(), foundDC);

    UnqualifiedId FuncId;
    if (!ParseWithShortcuts(foundDC,SS,funcName,P,S,FuncId)) {
      // Failed parse, cleanup.
      // Destroy the scope we created first, and
      // restore the original.
      S.ExitDeclaratorContext(P.getCurScope());
      P.ExitScope();
      P.ExitScope();
      P.getCurScope()->setEntity(OldEntity);
      // Then exit.
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
      P.ExitScope();
      P.getCurScope()->setEntity(OldEntity);
      // Then cleanup and exit.
      return TheDecl;
    }

    //
    //  Destroy the scope we created, and restore the original.
    //
    S.ExitDeclaratorContext(P.getCurScope());
    P.ExitScope();
    P.ExitScope();
    P.getCurScope()->setEntity(OldEntity);
    //
    //  Check for lookup failure.
    //
    if (Result.getResultKind() != LookupResult::Found &&
        Result.getResultKind() != LookupResult::FoundOverloaded) {
       // Lookup failed.
       return TheDecl;
    }
    return functionSelector(foundDC,objectIsConst,GivenArgs,
                            Result,
                            FuncNameInfo,
                            FuncTemplateArgs,
                            Context, P, S);
  }

  static
  bool ParseProto(llvm::SmallVector<Expr*, 4> &GivenArgs,
                  ASTContext& Context, Parser &P,Sema &S) {
    //
    //  Parse the prototype now.
    //

    while (P.getCurToken().isNot(tok::eof)) {
      TypeResult Res(P.ParseTypeName());
      if (!Res.isUsable()) {
        // Bad parse, done.
        return false;
      }
      TypeSourceInfo *TSI = 0;
      clang::QualType QT = clang::Sema::GetTypeFromParser(Res.get(), &TSI);
      QT = QT.getCanonicalType();
      {
        ExprValueKind VK = VK_RValue;
        if (QT->getAs<LValueReferenceType>()) {
          VK = VK_LValue;
        }
        clang::QualType NonRefQT(QT.getNonReferenceType());
        Expr* val
          = new (Context) OpaqueValueExpr(TSI->getTypeLoc().getLocStart(),
                                          NonRefQT, VK);
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
      return false;
    }
    //
    //  Cleanup after prototype parse.
    //
    P.SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
                /*StopAtCodeCompletion*/false);
    S.getDiagnostics().Reset();

    return true;
  }

  const FunctionDecl* LookupHelper::findFunctionProto(const Decl* scopeDecl,
                                                      llvm::StringRef funcName,
                                                      llvm::StringRef funcProto,
                                                      bool objectIsConst
                                                      ) const {
    assert(scopeDecl && "Decl cannot be null");
    //
    //  Some utilities.
    //
    // Use P for shortness
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    ASTContext& Context = S.getASTContext();

    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
    //
    //  Do this 'early' to save on the expansive parser setup,
    //  in case of failure.
    //
    CXXScopeSpec SS;
    DeclContext* foundDC = getContextAndSpec(SS,scopeDecl,Context,S);
    if (!foundDC) return 0;

    //
    //  Parse the prototype now.
    //
    ParserStateRAII ResetParserState(P);
    prepareForParsing(funcProto, llvm::StringRef("func.prototype.file"));

    llvm::SmallVector<Expr*, 4> GivenArgs;
    if (!funcProto.empty()) {
      if (!ParseProto(GivenArgs,Context,P,S) ) {
        return 0;
      }
    }

    Interpreter::PushTransactionRAII pushedT(m_Interpreter);
    return findFunction(foundDC, SS,
                        funcName, GivenArgs, objectIsConst,
                        Context, P, S,
                        overloadFunctionSelector);
  }

  const FunctionDecl* LookupHelper::matchFunctionProto(const Decl* scopeDecl,
                                                       llvm::StringRef funcName,
                                                       llvm::StringRef funcProto,
                                                       bool objectIsConst
                                                       ) const {
    assert(scopeDecl && "Decl cannot be null");
    //
    //  Some utilities.
    //
    // Use P for shortness
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    ASTContext& Context = S.getASTContext();

    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
    //
    //  Do this 'early' to save on the expansive parser setup,
    //  in case of failure.
    //
    CXXScopeSpec SS;
    DeclContext* foundDC = getContextAndSpec(SS,scopeDecl,Context,S);
    if (!foundDC) return 0;

    //
    //  Parse the prototype now.
    //
    ParserStateRAII ResetParserState(P);
    prepareForParsing(funcProto, llvm::StringRef("func.prototype.file"));

    llvm::SmallVector<Expr*, 4> GivenArgs;
    if (!funcProto.empty()) {
      if (!ParseProto(GivenArgs,Context,P,S) ) {
        return 0;
      }
    }

    Interpreter::PushTransactionRAII pushedT(m_Interpreter);
    return findFunction(foundDC, SS,
                        funcName, GivenArgs, objectIsConst,
                        Context, P, S,
                        matchFunctionSelector);
  }

  static
  bool ParseArgs(llvm::SmallVector<Expr*, 4> &GivenArgs,
                 ASTContext& Context, Parser &P, Sema &S) {

    //
    //  Parse the arguments now.
    //

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
    // For backward compatibility with CINT accept (for now?) a trailing close
    // parenthesis.
    if (P.getCurToken().isNot(tok::eof) && P.getCurToken().isNot(tok::r_paren) ) {
      // We did not consume all of the arg list, bad parse.
      return false;
    }
    //
    //  Cleanup after the arg list parse.
    //
    P.SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
                /*StopAtCodeCompletion*/false);
    S.getDiagnostics().Reset();
    return true;
  }

  const FunctionDecl* LookupHelper::findFunctionArgs(const Decl* scopeDecl,
                                                     llvm::StringRef funcName,
                                                     llvm::StringRef funcArgs,
                                                     bool objectIsConst
                                                     ) const {
    assert(scopeDecl && "Decl cannot be null");
    //
    //  Some utilities.
    //
    // Use P for shortness
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    ASTContext& Context = S.getASTContext();

    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
    //
    //  Do this 'early' to save on the expansive parser setup,
    //  in case of failure.
    //
    CXXScopeSpec SS;
    DeclContext* foundDC = getContextAndSpec(SS,scopeDecl,Context,S);
    if (!foundDC) return 0;

    //
    //  Parse the arguments now.
    //
    ParserStateRAII ResetParserState(P);
    prepareForParsing(funcArgs, llvm::StringRef("func.args.file"));

    llvm::SmallVector<Expr*, 4> GivenArgs;
    if (!funcArgs.empty()) {
      if (!ParseArgs(GivenArgs,Context,P,S) ) {
        return 0;
      }
    }

    Interpreter::PushTransactionRAII pushedT(m_Interpreter);
    return findFunction(foundDC, SS,
                        funcName, GivenArgs, objectIsConst,
                        Context, P, S, overloadFunctionSelector);
  }

  void LookupHelper::findArgList(llvm::StringRef argList,
                                 llvm::SmallVector<Expr*, 4>& argExprs) const {
    if (argList.empty()) return;

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
    //  Turn on ignoring of the main file eof token.
    //
    //  Note: We need this because token readahead in the following
    //        routine calls ends up parsing it multiple times.
    //
    if (!PP.isIncrementalProcessingEnabled()) {
      PP.enableIncrementalProcessing();
    }
    if (!code.empty()) {
      //
      //  Create a fake file to parse the type name.
      //
      llvm::MemoryBuffer* SB
         = llvm::MemoryBuffer::getMemBufferCopy(code.str() + "\n",
                                                bufferName.str());
      FileID FID = S.getSourceManager().createFileIDForMemBuffer(SB);
      //
      //  Switch to the new file the way #include does.
      //
      //  Note: To switch back to the main file we must consume an eof token.
      //
      PP.EnterSourceFile(FID, /*DirLookup=*/0, SourceLocation());
      PP.Lex(const_cast<Token&>(P.getCurToken()));
    }
  }

  static
  bool hasFunctionSelector(DeclContext* ,
                           bool /* objectIsConst */,
                           const llvm::SmallVector<Expr*, 4> &,
                           LookupResult &Result,
                           DeclarationNameInfo &,
                           const TemplateArgumentListInfo* ,
                           ASTContext&, Parser &, Sema &) {
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

  bool LookupHelper::hasFunction(const clang::Decl* scopeDecl,
                                 llvm::StringRef funcName) const {

    //FIXME: remore code duplication with findFunctionArgs() and friends.

    assert(scopeDecl && "Decl cannot be null");
    //
    //  Some utilities.
    //
    Parser& P = *m_Parser;
    Sema& S = P.getActions();
    ASTContext& Context = S.getASTContext();

    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
    //
    //  Do this 'early' to save on the expansive parser setup,
    //  in case of failure.
    //
    CXXScopeSpec SS;
    DeclContext* foundDC = getContextAndSpec(SS,scopeDecl,Context,S);
    if (!foundDC) return 0;

    ParserStateRAII ResetParserState(P);
    llvm::SmallVector<Expr*, 4> GivenArgs;

    Interpreter::PushTransactionRAII pushedT(m_Interpreter);
    return findFunction(foundDC, SS,
                        funcName, GivenArgs, false /* objectIsConst */,
                        Context, P, S, hasFunctionSelector);
  }


} // end namespace cling
