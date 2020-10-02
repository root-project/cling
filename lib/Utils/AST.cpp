//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/DeclTemplate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "clang/AST/Mangle.h"

#include <memory>
#include <stdio.h>

using namespace clang;

namespace {
  template<typename D>
  static D* LookupResult2Decl(clang::LookupResult& R)
  {
    if (R.empty())
      return 0;

    R.resolveKind();

    if (R.isSingleResult())
      return dyn_cast<D>(R.getFoundDecl());
    return (D*)-1;
  }
}

namespace cling {
namespace utils {

  static
  QualType GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx,
                                         QualType QT,
                               const Transform::Config& TypeConfig,
                                         bool fullyQualifyType,
                                         bool fullyQualifyTmpltArg);

  static
  NestedNameSpecifier* GetPartiallyDesugaredNNS(const ASTContext& Ctx,
                                                NestedNameSpecifier* scope,
                                           const Transform::Config& TypeConfig);

  static NestedNameSpecifier*
  CreateNestedNameSpecifierForScopeOf(const ASTContext& Ctx,
                                      const Decl *decl,
                                      bool FullyQualified);

  static
  NestedNameSpecifier* GetFullyQualifiedNameSpecifier(const ASTContext& Ctx,
                                                      NestedNameSpecifier* scope);

  bool Analyze::IsWrapper(const FunctionDecl* ND) {
    if (!ND)
      return false;
    if (!ND->getDeclName().isIdentifier())
      return false;

    return ND->getName().startswith(Synthesize::UniquePrefix);
  }

  void Analyze::maybeMangleDeclName(const GlobalDecl& GD,
                                    std::string& mangledName) {
    // copied and adapted from CodeGen::CodeGenModule::getMangledName

    NamedDecl* D
      = cast<NamedDecl>(const_cast<Decl*>(GD.getDecl()));
    std::unique_ptr<MangleContext> mangleCtx;
    mangleCtx.reset(D->getASTContext().createMangleContext());
    if (!mangleCtx->shouldMangleDeclName(D)) {
      IdentifierInfo *II = D->getIdentifier();
      assert(II && "Attempt to mangle unnamed decl.");
      mangledName = II->getName();
      return;
    }

    llvm::raw_string_ostream RawStr(mangledName);
    switch(D->getKind()) {
    case Decl::CXXConstructor:
      //Ctor_Complete,          // Complete object ctor
      //Ctor_Base,              // Base object ctor
      //Ctor_CompleteAllocating // Complete object allocating ctor (unused)
      mangleCtx->mangleCXXCtor(cast<CXXConstructorDecl>(D),
                               GD.getCtorType(), RawStr);
      break;

    case Decl::CXXDestructor:
      //Dtor_Deleting, // Deleting dtor
      //Dtor_Complete, // Complete object dtor
      //Dtor_Base      // Base object dtor
#if defined(LLVM_ON_WIN32)
      // MicrosoftMangle.cpp:954 calls llvm_unreachable when mangling Dtor_Comdat
      if (GD.getDtorType() == Dtor_Comdat) {
        if (const IdentifierInfo* II = D->getIdentifier())
          RawStr << II->getName();
      } else
#endif
      {
        mangleCtx->mangleCXXDtor(cast<CXXDestructorDecl>(D),
                                 GD.getDtorType(), RawStr);
      }
      break;

    default :
      mangleCtx->mangleName(D, RawStr);
      break;
    }
    RawStr.flush();
  }

  Expr* Analyze::GetOrCreateLastExpr(FunctionDecl* FD,
                                     int* FoundAt /*=0*/,
                                     bool omitDeclStmts /*=true*/,
                                     Sema* S /*=0*/) {
    assert(FD && "We need a function declaration!");
    assert((omitDeclStmts || S)
           && "Sema needs to be set when omitDeclStmts is false");
    if (FoundAt)
      *FoundAt = -1;

    Expr* result = 0;
    if (CompoundStmt* CS = dyn_cast<CompoundStmt>(FD->getBody())) {
      ArrayRef<Stmt*> Stmts
        = llvm::makeArrayRef(CS->body_begin(), CS->size());
      int indexOfLastExpr = Stmts.size();
      while(indexOfLastExpr--) {
        if (!isa<NullStmt>(Stmts[indexOfLastExpr]))
          break;
      }

      if (FoundAt)
        *FoundAt = indexOfLastExpr;

      if (indexOfLastExpr < 0)
        return 0;

      if ( (result = dyn_cast<Expr>(Stmts[indexOfLastExpr])) )
        return result;
      if (!omitDeclStmts)
        if (DeclStmt* DS = dyn_cast<DeclStmt>(Stmts[indexOfLastExpr])) {
          std::vector<Stmt*> newBody = Stmts.vec();
          for (DeclStmt::reverse_decl_iterator I = DS->decl_rbegin(),
                 E = DS->decl_rend(); I != E; ++I) {
            if (VarDecl* VD = dyn_cast<VarDecl>(*I)) {
              // Change the void function's return type
              // We can't PushDeclContext, because we don't have scope.
              Sema::ContextRAII pushedDC(*S, FD);

              QualType VDTy = VD->getType().getNonReferenceType();
              // Get the location of the place we will insert.
              SourceLocation Loc
                = newBody[indexOfLastExpr]->getLocEnd().getLocWithOffset(1);
              Expr* DRE = S->BuildDeclRefExpr(VD, VDTy,VK_LValue, Loc).get();
              assert(DRE && "Cannot be null");
              indexOfLastExpr++;
              newBody.insert(newBody.begin() + indexOfLastExpr, DRE);

              // Attach the new body (note: it does dealloc/alloc of all nodes)
              CS->setStmts(S->getASTContext(), newBody);
              if (FoundAt)
                *FoundAt = indexOfLastExpr;
              return DRE;
            }
          }
        }

      return result;
    }

    return result;
  }

  const char* const Synthesize::UniquePrefix = "__cling_Un1Qu3";

  IntegerLiteral* Synthesize::IntegerLiteralExpr(ASTContext& C, uintptr_t Ptr) {
    const llvm::APInt Addr(8 * sizeof(void*), Ptr);
    return IntegerLiteral::Create(C, Addr, C.getUIntPtrType(),
                                  SourceLocation());
  }

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, uintptr_t Ptr) {
    ASTContext& Ctx = S->getASTContext();
    return CStyleCastPtrExpr(S, Ty, Synthesize::IntegerLiteralExpr(Ctx, Ptr));
  }

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, Expr* E) {
    ASTContext& Ctx = S->getASTContext();
    if (!Ty->isPointerType())
      Ty = Ctx.getPointerType(Ty);

    TypeSourceInfo* TSI = Ctx.getTrivialTypeSourceInfo(Ty, SourceLocation());
    Expr* Result
      = S->BuildCStyleCastExpr(SourceLocation(), TSI,SourceLocation(),E).get();
    assert(Result && "Cannot create CStyleCastPtrExpr");
    return Result;
  }

  static bool
  GetFullyQualifiedTemplateName(const ASTContext& Ctx, TemplateName &tname) {

    bool changed = false;
    NestedNameSpecifier *NNS = 0;

    TemplateDecl *argtdecl = tname.getAsTemplateDecl();
    QualifiedTemplateName *qtname = tname.getAsQualifiedTemplateName();

    if (qtname && !qtname->hasTemplateKeyword()) {
      NNS = qtname->getQualifier();
      NestedNameSpecifier *qNNS = GetFullyQualifiedNameSpecifier(Ctx,NNS);
      if (qNNS != NNS) {
        changed = true;
        NNS = qNNS;
      } else {
        NNS = 0;
      }
    } else {
      NNS = CreateNestedNameSpecifierForScopeOf(Ctx, argtdecl, true);
    }
    if (NNS) {
      tname = Ctx.getQualifiedTemplateName(NNS,
                                           /*TemplateKeyword=*/ false,
                                           argtdecl);
      changed = true;
    }
    return changed;
  }

  static bool
  GetFullyQualifiedTemplateArgument(const ASTContext& Ctx,
                                    TemplateArgument &arg) {
    bool changed = false;

    // Note: we do not handle TemplateArgument::Expression, to replace it
    // we need the information for the template instance decl.
    // See GetPartiallyDesugaredTypeImpl

    if (arg.getKind() == TemplateArgument::Template) {
      TemplateName tname = arg.getAsTemplate();
      changed = GetFullyQualifiedTemplateName(Ctx, tname);
      if (changed) {
        arg = TemplateArgument(tname);
      }
    } else if (arg.getKind() == TemplateArgument::Type) {
      QualType SubTy = arg.getAsType();
      // Check if the type needs more desugaring and recurse.
      QualType QTFQ = TypeName::GetFullyQualifiedType(SubTy, Ctx);
      if (QTFQ != SubTy) {
        arg = TemplateArgument(QTFQ);
        changed = true;
      }
    } else if (arg.getKind() == TemplateArgument::Pack) {
      SmallVector<TemplateArgument, 2> desArgs;
      for (auto I = arg.pack_begin(), E = arg.pack_end(); I != E; ++I) {
        TemplateArgument pack_arg(*I);
        changed = GetFullyQualifiedTemplateArgument(Ctx,pack_arg);
        desArgs.push_back(pack_arg);
      }
      if (changed) {
        // The allocator in ASTContext is mutable ...
        // Keep the argument const to be inline will all the other interfaces
        // like:  NestedNameSpecifier::Create
        ASTContext &mutableCtx( const_cast<ASTContext&>(Ctx) );
        arg = TemplateArgument::CreatePackCopy(mutableCtx, desArgs);
      }
    }
    return changed;
  }

  static const Type*
  GetFullyQualifiedLocalType(const ASTContext& Ctx,
                             const Type *typeptr) {
    // We really just want to handle the template parameter if any ....
    // In case of template specializations iterate over the arguments and
    // fully qualify them as well.
    if (const TemplateSpecializationType* TST
        = llvm::dyn_cast<const TemplateSpecializationType>(typeptr)) {

      bool mightHaveChanged = false;
      llvm::SmallVector<TemplateArgument, 4> desArgs;
      for (TemplateSpecializationType::iterator
             I = TST->begin(), E = TST->end();
          I != E; ++I) {

        // cheap to copy and potentially modified by
        // GetFullyQualifedTemplateArgument
        TemplateArgument arg(*I);
        mightHaveChanged |= GetFullyQualifiedTemplateArgument(Ctx,arg);
        desArgs.push_back(arg);
      }

      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
        QualType QT
          = Ctx.getTemplateSpecializationType(TST->getTemplateName(),
                                              desArgs,
                                              TST->getCanonicalTypeInternal());
        return QT.getTypePtr();
      }
    } else if (const RecordType *TSTRecord
               = llvm::dyn_cast<const RecordType>(typeptr)) {
      // We are asked to fully qualify and we have a Record Type,
      // which can point to a template instantiation with no sugar in any of
      // its template argument, however we still need to fully qualify them.

      if (const ClassTemplateSpecializationDecl* TSTdecl =
          llvm::dyn_cast<ClassTemplateSpecializationDecl>(TSTRecord->getDecl()))
        {
          const TemplateArgumentList& templateArgs
            = TSTdecl->getTemplateArgs();

          bool mightHaveChanged = false;
          llvm::SmallVector<TemplateArgument, 4> desArgs;
          for(unsigned int I = 0, E = templateArgs.size();
              I != E; ++I) {

            // cheap to copy and potentially modified by
            // GetFullyQualifedTemplateArgument
            TemplateArgument arg(templateArgs[I]);
            mightHaveChanged |= GetFullyQualifiedTemplateArgument(Ctx,arg);
            desArgs.push_back(arg);

          }

          // If desugaring happened allocate new type in the AST.
          if (mightHaveChanged) {
            TemplateName TN(TSTdecl->getSpecializedTemplate());
            QualType QT
              = Ctx.getTemplateSpecializationType(TN, desArgs,
                                         TSTRecord->getCanonicalTypeInternal());
            return QT.getTypePtr();
          }
        }
    }
    return typeptr;
  }

  static NestedNameSpecifier* CreateOuterNNS(const ASTContext& Ctx,
                                             const Decl* D,
                                             bool FullyQualify) {
    const DeclContext* DC = D->getDeclContext();
    if (const NamespaceDecl* NS = dyn_cast<NamespaceDecl>(DC)) {
      while (NS && NS->isInline()) {
        // Ignore inline namespace;
        NS = dyn_cast_or_null<NamespaceDecl>(NS->getDeclContext());
      }
      if (NS && NS->getDeclName())
        return TypeName::CreateNestedNameSpecifier(Ctx, NS);
      return nullptr; // no starting '::', no anonymous
    } else if (const TagDecl* TD = dyn_cast<TagDecl>(DC)) {
      return TypeName::CreateNestedNameSpecifier(Ctx, TD, FullyQualify);
    } else if (const TypedefNameDecl* TDD = dyn_cast<TypedefNameDecl>(DC)) {
      return TypeName::CreateNestedNameSpecifier(Ctx, TDD, FullyQualify);
    }
    return nullptr; // no starting '::'
  }

  static
  NestedNameSpecifier* GetFullyQualifiedNameSpecifier(const ASTContext& Ctx,
                                                  NestedNameSpecifier* scope) {
    // Return a fully qualified version of this name specifier
    if (scope->getKind() == NestedNameSpecifier::Global) {
      // Already fully qualified.
      return scope;
    }

    if (const Type *type = scope->getAsType()) {
      // Find decl context.
      const TagDecl* TD = 0;
      if (const TagType* tagdecltype = dyn_cast<TagType>(type)) {
        TD = tagdecltype->getDecl();
      } else {
        TD = type->getAsCXXRecordDecl();
      }
      if (TD) {
        return TypeName::CreateNestedNameSpecifier(Ctx, TD,
                                                   true /*FullyQualified*/);
      } else if (const TypedefType* TDD = dyn_cast<TypedefType>(type)) {
        return TypeName::CreateNestedNameSpecifier(Ctx, TDD->getDecl(),
                                                   true /*FullyQualified*/);
      }
    } else if (const NamespaceDecl* NS = scope->getAsNamespace()) {
      return TypeName::CreateNestedNameSpecifier(Ctx, NS);
    } else if (const NamespaceAliasDecl* alias = scope->getAsNamespaceAlias()) {
      const NamespaceDecl* NS = alias->getNamespace()->getCanonicalDecl();
      return TypeName::CreateNestedNameSpecifier(Ctx, NS);
    }

    return scope;
  }

  static
  NestedNameSpecifier* SelectPrefix(const ASTContext& Ctx,
                                    const DeclContext *declContext,
                                    NestedNameSpecifier *original_prefix,
                             const Transform::Config& TypeConfig) {
    // We have to also desugar the prefix.
    NestedNameSpecifier* prefix = 0;
    if (declContext) {
      // We had a scope prefix as input, let see if it is still
      // the same as the scope of the result and if it is, then
      // we use it.
      if (declContext->isNamespace()) {
        // Deal with namespace.  This is mostly about dealing with
        // namespace aliases (i.e. keeping the one the user used).
        const NamespaceDecl *new_ns =dyn_cast<NamespaceDecl>(declContext);
        if (new_ns) {
          new_ns = new_ns->getCanonicalDecl();
          NamespaceDecl *old_ns = 0;
          if (original_prefix) {
            original_prefix->getAsNamespace();
            if (NamespaceAliasDecl *alias =
                     original_prefix->getAsNamespaceAlias())
            {
              old_ns = alias->getNamespace()->getCanonicalDecl();
            }
          }
          if (old_ns == new_ns) {
            // This is the same namespace, use the original prefix
            // as a starting point.
            prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
          } else {
            prefix = TypeName::CreateNestedNameSpecifier(Ctx,
                                               dyn_cast<NamespaceDecl>(new_ns));
          }
        }
      } else {
        const CXXRecordDecl* newtype=dyn_cast<CXXRecordDecl>(declContext);
        if (newtype && original_prefix) {
          // Deal with a class
          const Type *oldtype = original_prefix->getAsType();
          if (oldtype &&
              // NOTE: Should we compare the RecordDecl instead?
              oldtype->getAsCXXRecordDecl() == newtype)
          {
            // This is the same type, use the original prefix as a starting
            // point.
            prefix = GetPartiallyDesugaredNNS(Ctx,original_prefix,TypeConfig);
          } else {
            const TagDecl *tdecl = dyn_cast<TagDecl>(declContext);
            if (tdecl) {
              prefix = TypeName::CreateNestedNameSpecifier(Ctx, tdecl,
                                                      false /*FullyQualified*/);
            }
          }
        } else {
          // We should only create the nested name specifier
          // if the outer scope is really a TagDecl.
          // It could also be a CXXMethod for example.
          const TagDecl *tdecl = dyn_cast<TagDecl>(declContext);
          if (tdecl) {
            prefix = TypeName::CreateNestedNameSpecifier(Ctx,tdecl,
                                                      false /*FullyQualified*/);
          }
        }
      }
    } else {
      prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
    }
    return prefix;
  }

  static
  NestedNameSpecifier* SelectPrefix(const ASTContext& Ctx,
                                    const ElaboratedType *etype,
                                    NestedNameSpecifier *original_prefix,
                             const Transform::Config& TypeConfig) {
    // We have to also desugar the prefix.

    NestedNameSpecifier* prefix = etype->getQualifier();
    if (original_prefix && prefix) {
      // We had a scope prefix as input, let see if it is still
      // the same as the scope of the result and if it is, then
      // we use it.
      const Type *newtype = prefix->getAsType();
      if (newtype) {
        // Deal with a class
        const Type *oldtype = original_prefix->getAsType();
        if (oldtype &&
            // NOTE: Should we compare the RecordDecl instead?
            oldtype->getAsCXXRecordDecl() == newtype->getAsCXXRecordDecl())
        {
          // This is the same type, use the original prefix as a starting
          // point.
          prefix = GetPartiallyDesugaredNNS(Ctx,original_prefix,TypeConfig);
        } else {
          prefix = GetPartiallyDesugaredNNS(Ctx,prefix,TypeConfig);
        }
      } else {
        // Deal with namespace.  This is mostly about dealing with
        // namespace aliases (i.e. keeping the one the user used).
        const NamespaceDecl *new_ns = prefix->getAsNamespace();
        if (new_ns) {
          new_ns = new_ns->getCanonicalDecl();
        }
        else if (NamespaceAliasDecl *alias = prefix->getAsNamespaceAlias() )
        {
          new_ns = alias->getNamespace()->getCanonicalDecl();
        }
        if (new_ns) {
          const NamespaceDecl *old_ns = original_prefix->getAsNamespace();
          if (old_ns) {
            old_ns = old_ns->getCanonicalDecl();
          }
          else if (NamespaceAliasDecl *alias =
                   original_prefix->getAsNamespaceAlias())
          {
            old_ns = alias->getNamespace()->getCanonicalDecl();
          }
          if (old_ns == new_ns) {
            // This is the same namespace, use the original prefix
            // as a starting point.
            prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
          } else {
            prefix = GetFullyQualifiedNameSpecifier(Ctx,prefix);
          }
        } else {
          prefix = GetFullyQualifiedNameSpecifier(Ctx,prefix);
        }
      }
    }
    return prefix;
  }


  static
  NestedNameSpecifier* GetPartiallyDesugaredNNS(const ASTContext& Ctx,
                                                NestedNameSpecifier* scope,
                                          const Transform::Config& TypeConfig) {
    // Desugar the scope qualifier if needed.

    if (const Type* scope_type = scope->getAsType()) {

      // this is not a namespace, so we might need to desugar
      QualType desugared = GetPartiallyDesugaredTypeImpl(Ctx,
                                                         QualType(scope_type,0),
                                                         TypeConfig,
                                                         /*qualifyType=*/false,
                                                      /*qualifyTmpltArg=*/true);

      NestedNameSpecifier* outer_scope = scope->getPrefix();
      const ElaboratedType* etype
         = dyn_cast<ElaboratedType>(desugared.getTypePtr());
      if (etype) {
        // The desugarding returned an elaborated type even-though we
        // did not request it (/*fullyQualify=*/false), so we must have been
        // looking a typedef pointing at a (or another) scope.

        if (outer_scope) {
          outer_scope = SelectPrefix(Ctx,etype,outer_scope,TypeConfig);
        } else {
          outer_scope = GetPartiallyDesugaredNNS(Ctx,etype->getQualifier(),
                                                 TypeConfig);
        }
        desugared = etype->getNamedType();
      } else {

        Decl* decl = 0;
        const TypedefType* typedeftype =
          dyn_cast_or_null<TypedefType>(&(*desugared));
        if (typedeftype) {
          decl = typedeftype->getDecl();
        } else {
          // There are probably other cases ...
          const TagType* tagdecltype = dyn_cast_or_null<TagType>(&(*desugared));
          if (tagdecltype) {
            decl = tagdecltype->getDecl();
          } else {
            decl = desugared->getAsCXXRecordDecl();
          }
        }
        if (decl) {
          NamedDecl* outer
            = dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
          NamespaceDecl* outer_ns
            = dyn_cast_or_null<NamespaceDecl>(decl->getDeclContext());
          if (outer
              && !(outer_ns && outer_ns->isAnonymousNamespace())
              && outer->getName().size() ) {
            outer_scope = SelectPrefix(Ctx,decl->getDeclContext(),
                                       outer_scope,TypeConfig);
          } else {
            outer_scope = 0;
          }
        } else if (outer_scope) {
          outer_scope = GetPartiallyDesugaredNNS(Ctx, outer_scope, TypeConfig);
        }
      }
      return NestedNameSpecifier::Create(Ctx,outer_scope,
                                         false /* template keyword wanted */,
                                         desugared.getTypePtr());
    } else {
      return  GetFullyQualifiedNameSpecifier(Ctx,scope);
    }
  }

  bool Analyze::IsStdOrCompilerDetails(const NamedDecl &decl)
  {
    // Return true if the TagType is a 'details' of the std implementation
    // or declared within std.
    // Details means (For now) declared in __gnu_cxx or starting with
    // underscore.

    IdentifierInfo *info = decl.getDeclName().getAsIdentifierInfo();
    if (info && info->getNameStart()[0] == '_') {
      // We have a name starting by _, this is reserve for compiler
      // implementation, so let's not desugar to it.
      return true;
    }
    // And let's check if it is in one of the know compiler implementation
    // namespace.
    const NamedDecl *outer =dyn_cast_or_null<NamedDecl>(decl.getDeclContext());
    while (outer && outer->getName().size() ) {
      if (outer->getName().compare("std") == 0 ||
          outer->getName().compare("__gnu_cxx") == 0) {
        return true;
      }
      outer = dyn_cast_or_null<NamedDecl>(outer->getDeclContext());
    }
    return false;
  }

  bool Analyze::IsStdClass(const clang::NamedDecl &cl)
  {
    // Return true if the class or template is declared directly in the
    // std namespace (modulo inline namespace).

    return cl.getDeclContext()->isStdNamespace();
  }

  // See Sema::PushOnScopeChains
  bool Analyze::isOnScopeChains(const NamedDecl* ND, Sema& SemaR) {

    // Named decls without name shouldn't be in. Eg: struct {int a};
    if (!ND->getDeclName())
      return false;

    // Out-of-line definitions shouldn't be pushed into scope in C++.
    // Out-of-line variable and function definitions shouldn't even in C.
    if ((isa<VarDecl>(ND) || isa<FunctionDecl>(ND)) && ND->isOutOfLine() &&
        !ND->getDeclContext()->getRedeclContext()->Equals(
                        ND->getLexicalDeclContext()->getRedeclContext()))
      return false;

    // Template instantiations should also not be pushed into scope.
    if (isa<FunctionDecl>(ND) &&
        cast<FunctionDecl>(ND)->isFunctionTemplateSpecialization())
      return false;

    // Using directives are not registered onto the scope chain
    if (isa<UsingDirectiveDecl>(ND))
      return false;

    IdentifierResolver::iterator
      IDRi = SemaR.IdResolver.begin(ND->getDeclName()),
      IDRiEnd = SemaR.IdResolver.end();

    for (; IDRi != IDRiEnd; ++IDRi) {
      if (ND == *IDRi)
        return true;
    }


    // Check if the declaration is template instantiation, which is not in
    // any DeclContext yet, because it came from
    // Sema::PerformPendingInstantiations
    // if (isa<FunctionDecl>(D) &&
    //     cast<FunctionDecl>(D)->getTemplateInstantiationPattern())
    //   return false;


    return false;
  }

  unsigned int
  Transform::Config::DropDefaultArg(clang::TemplateDecl &Template) const
  {
    /// Return the number of default argument to drop.

    if (Analyze::IsStdClass(Template)) {
      static const char *stls[] =  //container names
        {"vector","list","deque","map","multimap","set","multiset",0};
      static unsigned int values[] =       //number of default arg.
        {1,1,1,2,2,2,2};
      StringRef name = Template.getName();
      for(int k=0;stls[k];k++) {
        if ( name.equals(stls[k]) ) return values[k];
      }
    }
    // Check in some struct if the Template decl is registered something like
    /*
     DefaultCollection::const_iterator iter;
     iter = m_defaultArgs.find(&Template);
     if (iter != m_defaultArgs.end()) {
        return iter->second;
     }
    */
    return 0;
  }

  static bool ShouldKeepTypedef(const TypedefType* TT,
                                const llvm::SmallSet<const Decl*, 4>& ToSkip)
  {
    // Return true, if we should keep this typedef rather than desugaring it.

    return 0 != ToSkip.count(TT->getDecl()->getCanonicalDecl());
  }

  static bool SingleStepPartiallyDesugarTypeImpl(QualType& QT)
  {
    //  WARNING:
    //
    //  The large blocks of commented-out code in this routine
    //  are there to support doing more desugaring in the future,
    //  we will probably have to.
    //
    //  Do not delete until we are completely sure we will
    //  not be changing this routine again!
    //
    const Type* QTy = QT.getTypePtr();
    Type::TypeClass TC = QTy->getTypeClass();
    switch (TC) {
      //
      //  Unconditionally sugared types.
      //
      case Type::Paren: {
        return false;
        //const ParenType* Ty = llvm::cast<ParenType>(QTy);
        //QT = Ty->desugar();
        //return true;
      }
      case Type::Typedef: {
        const TypedefType* Ty = llvm::cast<TypedefType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::TypeOf: {
        const TypeOfType* Ty = llvm::cast<TypeOfType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::Attributed: {
        return false;
        //const AttributedType* Ty = llvm::cast<AttributedType>(QTy);
        //QT = Ty->desugar();
        //return true;
      }
      case Type::SubstTemplateTypeParm: {
        const SubstTemplateTypeParmType* Ty =
          llvm::cast<SubstTemplateTypeParmType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::Elaborated: {
        const ElaboratedType* Ty = llvm::cast<ElaboratedType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      //
      //  Conditionally sugared types.
      //
      case Type::TypeOfExpr: {
        const TypeOfExprType* Ty = llvm::cast<TypeOfExprType>(QTy);
        if (Ty->isSugared()) {
          QT = Ty->desugar();
          return true;
        }
        return false;
      }
      case Type::Decltype: {
        const DecltypeType* Ty = llvm::cast<DecltypeType>(QTy);
        if (Ty->isSugared()) {
          QT = Ty->desugar();
          return true;
        }
        return false;
      }
      case Type::UnaryTransform: {
        return false;
        //const UnaryTransformType* Ty = llvm::cast<UnaryTransformType>(QTy);
        //if (Ty->isSugared()) {
        //  QT = Ty->desugar();
        //  return true;
        //}
        //return false;
      }
      case Type::Auto: {
        return false;
        //const AutoType* Ty = llvm::cast<AutoType>(QTy);
        //if (Ty->isSugared()) {
        //  QT = Ty->desugar();
        //  return true;
        //}
        //return false;
      }
      case Type::TemplateSpecialization: {
        //const TemplateSpecializationType* Ty =
        //  llvm::cast<TemplateSpecializationType>(QTy);
        // Too broad, this returns a the target template but with
        // canonical argument types.
        //if (Ty->isTypeAlias()) {
        //  QT = Ty->getAliasedType();
        //  return true;
        //}
        // Too broad, this returns the canonical type
        //if (Ty->isSugared()) {
        //  QT = Ty->desugar();
        //  return true;
        //}
        return false;
      }
      // Not a sugared type.
      default: {
        break;
      }
    }
    return false;
  }

  bool Transform::SingleStepPartiallyDesugarType(QualType &QT,
                                                 const ASTContext &Context) {
    Qualifiers quals = QT.getQualifiers();
    bool desugared = SingleStepPartiallyDesugarTypeImpl( QT );
    if (desugared) {
      // If the types has been desugared it also lost its qualifiers.
      QT = Context.getQualifiedType(QT, quals);
    }
    return desugared;
  }

  static bool GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx,
                                            TemplateArgument &arg,
                                            const Transform::Config& TypeConfig,
                                            bool fullyQualifyTmpltArg) {
    bool changed = false;

    if (arg.getKind() == TemplateArgument::Template) {
      TemplateName tname = arg.getAsTemplate();
      // Note: should we not also desugar?
      changed = GetFullyQualifiedTemplateName(Ctx, tname);
      if (changed) {
        arg = TemplateArgument(tname);
      }
    } else if (arg.getKind() == TemplateArgument::Type) {

      QualType SubTy = arg.getAsType();
      // Check if the type needs more desugaring and recurse.
      if (isa<TypedefType>(SubTy)
          || isa<TemplateSpecializationType>(SubTy)
          || isa<ElaboratedType>(SubTy)
          || fullyQualifyTmpltArg) {
        changed = true;
        QualType PDQT
              = GetPartiallyDesugaredTypeImpl(Ctx, SubTy, TypeConfig,
                    /*fullyQualifyType=*/true,
                    /*fullyQualifyTmpltArg=*/true);
        arg = TemplateArgument(PDQT);
      }
    } else if (arg.getKind() == TemplateArgument::Pack) {
      SmallVector<TemplateArgument, 2> desArgs;
      for (auto I = arg.pack_begin(), E = arg.pack_end(); I != E; ++I) {
        TemplateArgument pack_arg(*I);
        changed = GetPartiallyDesugaredTypeImpl(Ctx,pack_arg,
                                                TypeConfig,
                                                fullyQualifyTmpltArg);
        desArgs.push_back(pack_arg);
      }
      if (changed) {
        // The allocator in ASTContext is mutable ...
        // Keep the argument const to be inline will all the other interfaces
        // like:  NestedNameSpecifier::Create
        ASTContext &mutableCtx( const_cast<ASTContext&>(Ctx) );
        arg = TemplateArgument::CreatePackCopy(mutableCtx, desArgs);
      }
    }
    return changed;
  }

  static const TemplateArgument*
  GetTmpltArgDeepFirstIndexPack(size_t &cur,
                                const TemplateArgument& arg,
                                size_t idx) {
    SmallVector<TemplateArgument, 2> desArgs;
    for (auto I = arg.pack_begin(), E = arg.pack_end();
         cur < idx && I != E; ++cur,++I) {
      if ((*I).getKind() == TemplateArgument::Pack) {
        auto p_arg = GetTmpltArgDeepFirstIndexPack(cur,(*I),idx);
        if (cur == idx) return p_arg;
      } else if (cur == idx) {
        return I;
      }
    }
    return nullptr;
  }

  // Return the template argument corresponding to the index (idx)
  // when the composite list of arguement is seen flattened out deep
  // first (where depth is provided by template argument packs)
  static const TemplateArgument*
  GetTmpltArgDeepFirstIndex(const TemplateArgumentList& templateArgs,
                            size_t idx) {

    for (size_t cur = 0, I = 0, E = templateArgs.size();
         cur <= idx && I < E; ++I, ++cur) {
      auto &arg = templateArgs[I];
      if (arg.getKind() == TemplateArgument::Pack) {
        // Need to recurse.
        auto p_arg = GetTmpltArgDeepFirstIndexPack(cur,arg,idx);
        if (cur == idx) return p_arg;
     } else if (cur == idx) {
        return &arg;
      }
    }
    return nullptr;
  }

  static QualType GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx,
    QualType QT, const Transform::Config& TypeConfig,
    bool fullyQualifyType, bool fullyQualifyTmpltArg)
  {
    if (QT.isNull())
      return QT;
    // If there are no constraints, then use the standard desugaring.
    if (TypeConfig.empty() && !fullyQualifyType && !fullyQualifyTmpltArg)
      return QT.getDesugaredType(Ctx);

    // In case of Int_t* we need to strip the pointer first, desugar and attach
    // the pointer once again.
    if (isa<PointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();
      QualType nQT;
      nQT = GetPartiallyDesugaredTypeImpl(Ctx, QT->getPointeeType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
      if (nQT == QT->getPointeeType()) return QT;

      QT = Ctx.getPointerType(nQT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    while (isa<SubstTemplateTypeParmType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();

      QT = dyn_cast<SubstTemplateTypeParmType>(QT.getTypePtr())->desugar();

      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
    }

    // In case of Int_t& we need to strip the pointer first, desugar and attach
    // the reference once again.
    if (isa<ReferenceType>(QT.getTypePtr())) {
      // Get the qualifiers.
      bool isLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
      Qualifiers quals = QT.getQualifiers();
      QualType nQT;
      nQT = GetPartiallyDesugaredTypeImpl(Ctx, QT->getPointeeType(), TypeConfig,
                                         fullyQualifyType,fullyQualifyTmpltArg);
      if (nQT == QT->getPointeeType()) return QT;

      // Add the r- or l-value reference type back to the desugared one.
      if (isLValueRefTy)
        QT = Ctx.getLValueReferenceType(nQT);
      else
        QT = Ctx.getRValueReferenceType(nQT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // In case of Int_t[2] we need to strip the array first, desugar and attach
    // the array once again.
    if (isa<ArrayType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();

      if (isa<ConstantArrayType>(QT.getTypePtr())) {
        const ConstantArrayType *arr
          = dyn_cast<ConstantArrayType>(QT.getTypePtr());
        QualType newQT
           = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                         fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == arr->getElementType()) return QT;
        QT = Ctx.getConstantArrayType (newQT,
                                       arr->getSize(),
                                       arr->getSizeModifier(),
                                       arr->getIndexTypeCVRQualifiers());

      } else if (isa<DependentSizedArrayType>(QT.getTypePtr())) {
        const DependentSizedArrayType *arr
          = dyn_cast<DependentSizedArrayType>(QT.getTypePtr());
        QualType newQT
          = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == QT) return QT;
        QT = Ctx.getDependentSizedArrayType (newQT,
                                            arr->getSizeExpr(),
                                            arr->getSizeModifier(),
                                            arr->getIndexTypeCVRQualifiers(),
                                            arr->getBracketsRange());

      } else if (isa<IncompleteArrayType>(QT.getTypePtr())) {
        const IncompleteArrayType *arr
          = dyn_cast<IncompleteArrayType>(QT.getTypePtr());
        QualType newQT
          = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == arr->getElementType()) return QT;
        QT = Ctx.getIncompleteArrayType (newQT,
                                         arr->getSizeModifier(),
                                         arr->getIndexTypeCVRQualifiers());

      } else if (isa<VariableArrayType>(QT.getTypePtr())) {
        const VariableArrayType *arr
          = dyn_cast<VariableArrayType>(QT.getTypePtr());
        QualType newQT
          = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == arr->getElementType()) return QT;
        QT = Ctx.getVariableArrayType (newQT,
                                       arr->getSizeExpr(),
                                       arr->getSizeModifier(),
                                       arr->getIndexTypeCVRQualifiers(),
                                       arr->getBracketsRange());
      }

      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // If the type is elaborated, first remove the prefix and then
    // when we are done we will as needed add back the (new) prefix.
    // for example for std::vector<int>::iterator, we work on
    // just 'iterator' (which remember which scope its from)
    // and remove the typedef to get (for example),
    //   __gnu_cxx::__normal_iterator
    // which is *not* in the std::vector<int> scope and it is
    // the __gnu__cxx part we should use as the prefix.
    // NOTE: however we problably want to add the std::vector typedefs
    // to the list of things to skip!

    NestedNameSpecifier* original_prefix = 0;
    Qualifiers prefix_qualifiers;
    const ElaboratedType* etype_input
      = dyn_cast<ElaboratedType>(QT.getTypePtr());
    if (etype_input) {
      // Intentionally, we do not care about the other compononent of
      // the elaborated type (the keyword) as part of the partial
      // desugaring (and/or name normaliztation) is to remove it.
      original_prefix = etype_input->getQualifier();
      if (original_prefix) {
        const NamespaceDecl *ns = original_prefix->getAsNamespace();
        if (!(ns && ns->isAnonymousNamespace())) {
          // We have to also desugar the prefix unless
          // it does not have a name (anonymous namespaces).
          fullyQualifyType = true;
          prefix_qualifiers = QT.getLocalQualifiers();
          QT = QualType(etype_input->getNamedType().getTypePtr(),0);
        } else {
          original_prefix = 0;
        }
      }
    }

    // Desugar QT until we cannot desugar any more, or
    // we hit one of the special typedefs.
    while (1) {
      if (const TypedefType* TT = llvm::dyn_cast<TypedefType>(QT.getTypePtr())){
        if (ShouldKeepTypedef(TT, TypeConfig.m_toSkip)) {
          if (!fullyQualifyType && !fullyQualifyTmpltArg) {
            return QT;
          }
          // We might have stripped the namespace/scope part,
          // so we must go on to add it back.
          break;
        }
      }
      bool wasDesugared = Transform::SingleStepPartiallyDesugarType(QT,Ctx);

      // Did we get to a basic_string, let's get back to std::string
      Transform::Config::ReplaceCollection::const_iterator
      iter = TypeConfig.m_toReplace.find(QT->getCanonicalTypeInternal().getTypePtr());
      if (iter != TypeConfig.m_toReplace.end()) {
        Qualifiers quals = QT.getQualifiers();
        QT = QualType( iter->second, 0);
        QT = Ctx.getQualifiedType(QT,quals);
        break;
      }
      if (!wasDesugared) {
        // No more work to do, stop now.
        break;
      }
    }

    // If we have a reference, array or pointer we still need to
    // desugar what they point to.
    if (isa<PointerType>(QT.getTypePtr()) ||
        isa<ReferenceType>(QT.getTypePtr()) ||
        isa<ArrayType>(QT.getTypePtr())) {
      return GetPartiallyDesugaredTypeImpl(Ctx, QT, TypeConfig,
                                           fullyQualifyType,
                                           fullyQualifyTmpltArg);
    }

    NestedNameSpecifier* prefix = 0;
    const ElaboratedType* etype
      = dyn_cast<ElaboratedType>(QT.getTypePtr());
    if (etype) {

      prefix = SelectPrefix(Ctx,etype,original_prefix,TypeConfig);

      prefix_qualifiers.addQualifiers(QT.getLocalQualifiers());
      QT = QualType(etype->getNamedType().getTypePtr(),0);

    } else if (fullyQualifyType) {
      // Let's check whether this type should have been an elaborated type.
      // in which case we want to add it ... but we can't really preserve
      // the typedef in this case ...

      Decl *decl = 0;
      const TypedefType* typedeftype =
        dyn_cast_or_null<TypedefType>(QT.getTypePtr());
      if (typedeftype) {
        decl = typedeftype->getDecl();
      } else {
        // There are probably other cases ...
        const TagType* tagdecltype = dyn_cast_or_null<TagType>(QT.getTypePtr());
        if (tagdecltype) {
          decl = tagdecltype->getDecl();
        } else {
          decl = QT->getAsCXXRecordDecl();
        }
      }
      if (decl) {
        NamedDecl* outer
           = dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
        NamespaceDecl* outer_ns
           = dyn_cast_or_null<NamespaceDecl>(decl->getDeclContext());
        if (outer
            && !(outer_ns && outer_ns->isAnonymousNamespace())
            && !outer->getNameAsString().empty() ) {
          if (original_prefix) {
            const Type *oldtype = original_prefix->getAsType();
            if (oldtype) {
              if (oldtype->getAsCXXRecordDecl() == outer) {
                // Same type, use the original spelling
                prefix
                  = GetPartiallyDesugaredNNS(Ctx, original_prefix, TypeConfig);
                outer = 0; // Cancel the later creation.
              }
            } else {
              const NamespaceDecl *old_ns = original_prefix->getAsNamespace();
              if (old_ns) {
                old_ns = old_ns->getCanonicalDecl();
              }
              else if (NamespaceAliasDecl *alias =
                       original_prefix->getAsNamespaceAlias())
              {
                old_ns = alias->getNamespace()->getCanonicalDecl();
              }
              const NamespaceDecl *new_ns = dyn_cast<NamespaceDecl>(outer);
              if (new_ns) new_ns = new_ns->getCanonicalDecl();
              if (old_ns == new_ns) {
                // This is the same namespace, use the original prefix
                // as a starting point.
                prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
                outer = 0; // Cancel the later creation.
              }
            }
          } else { // if (!original_prefix)
            // move qualifiers on the outer type (avoid 'std::const string'!)
            prefix_qualifiers = QT.getLocalQualifiers();
            QT = QualType(QT.getTypePtr(),0);
          }
          if (outer) {
            if (decl->getDeclContext()->isNamespace()) {
              prefix = TypeName::CreateNestedNameSpecifier(Ctx,
                                                dyn_cast<NamespaceDecl>(outer));
            } else {
              // We should only create the nested name specifier
              // if the outer scope is really a TagDecl.
              // It could also be a CXXMethod for example.
              TagDecl *tdecl = dyn_cast<TagDecl>(outer);
              if (tdecl) {
                prefix = TypeName::CreateNestedNameSpecifier(Ctx,tdecl,
                                                      false /*FullyQualified*/);
                prefix = GetPartiallyDesugaredNNS(Ctx,prefix,TypeConfig);
              }
            }
          }
        }
      }
    }

    // In case of template specializations iterate over the arguments and
    // desugar them as well.
    if (const TemplateSpecializationType* TST
       = dyn_cast<const TemplateSpecializationType>(QT.getTypePtr())) {

      if (TST->isTypeAlias()) {
        QualType targetType = TST->getAliasedType();
        /*
        // We really need to find a way to propagate/keep the opaque typedef
        // that are available in TST to the aliased type.  We would need
        // to do something like:

        QualType targetType = TST->getAliasedType();
        QualType resubst = ReSubstTemplateArg(targetType,TST);
        return GetPartiallyDesugaredTypeImpl(Ctx, resubst, TypeConfig,
                                             fullyQualifyType,
                                             fullyQualifyTmpltArg);

        // But this is not quite right (ReSubstTemplateArg is from TMetaUtils)
        // as it does not resubst for

          template <typename T> using myvector = std::vector<T>;
          myvector<Double32_t> vd32d;

        // and does not work at all for

          template<class T> using ptr = T*;
          ptr<Double32_t> p2;

        // as the target is not a template.
        */
        // So for now just return move on with the least lose we can do
        return GetPartiallyDesugaredTypeImpl(Ctx, targetType, TypeConfig,
                                           fullyQualifyType,
                                           fullyQualifyTmpltArg);
      }

      bool mightHaveChanged = false;
      llvm::SmallVector<TemplateArgument, 4> desArgs;
      unsigned int argi = 0;
      for(TemplateSpecializationType::iterator I = TST->begin(), E = TST->end();
          I != E; ++I, ++argi) {

        if (I->getKind() == TemplateArgument::Expression) {
          // If we have an expression, we need to replace it / desugar it
          // as it could contain unqualifed (or partially qualified or
          // private) parts.

          QualType canon = QT->getCanonicalTypeInternal();
          const RecordType *TSTRecord
            = dyn_cast<const RecordType>(canon.getTypePtr());
          if (TSTRecord) {
            if (const ClassTemplateSpecializationDecl* TSTdecl =
               dyn_cast<ClassTemplateSpecializationDecl>(TSTRecord->getDecl()))
            {
              const TemplateArgumentList& templateArgs
                = TSTdecl->getTemplateArgs();

              mightHaveChanged = true;
              const TemplateArgument *match
                  = GetTmpltArgDeepFirstIndex(templateArgs,argi);
              if (match) desArgs.push_back(*match);
              continue;
            }
          }
        }

        if (I->getKind() == TemplateArgument::Template) {
          TemplateName tname = I->getAsTemplate();
          // Note: should we not also desugar?
          bool changed = GetFullyQualifiedTemplateName(Ctx, tname);
          if (changed) {
            desArgs.push_back(TemplateArgument(tname));
            mightHaveChanged = true;
          } else
            desArgs.push_back(*I);
          continue;
        }

        if (I->getKind() != TemplateArgument::Type) {
          desArgs.push_back(*I);
          continue;
        }

        QualType SubTy = I->getAsType();
        // Check if the type needs more desugaring and recurse.
        if (isa<TypedefType>(SubTy)
            || isa<TemplateSpecializationType>(SubTy)
            || isa<ElaboratedType>(SubTy)
            || fullyQualifyTmpltArg) {
          QualType PDQT
            = GetPartiallyDesugaredTypeImpl(Ctx, SubTy, TypeConfig,
                                            fullyQualifyType,
                                            fullyQualifyTmpltArg);
          mightHaveChanged |= (SubTy != PDQT);
          desArgs.push_back(TemplateArgument(PDQT));
        } else {
          desArgs.push_back(*I);
        }
      }

      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
        Qualifiers qualifiers = QT.getLocalQualifiers();
        QT = Ctx.getTemplateSpecializationType(TST->getTemplateName(),
                                               desArgs,
                                               TST->getCanonicalTypeInternal());
        QT = Ctx.getQualifiedType(QT, qualifiers);
      }
    } else if (fullyQualifyTmpltArg) {

      if (const RecordType *TSTRecord
          = dyn_cast<const RecordType>(QT.getTypePtr())) {
        // We are asked to fully qualify and we have a Record Type,
        // which can point to a template instantiation with no sugar in any of
        // its template argument, however we still need to fully qualify them.

        if (const ClassTemplateSpecializationDecl* TSTdecl =
            dyn_cast<ClassTemplateSpecializationDecl>(TSTRecord->getDecl()))
        {
          const TemplateArgumentList& templateArgs
            = TSTdecl->getTemplateArgs();

          bool mightHaveChanged = false;
          llvm::SmallVector<TemplateArgument, 4> desArgs;
          for(unsigned int I = 0, E = templateArgs.size();
              I != E; ++I) {

#if 1

            // cheap to copy and potentially modified by
            // GetPartiallyDesugaredTypeImpl
            TemplateArgument arg(templateArgs[I]);
            mightHaveChanged |= GetPartiallyDesugaredTypeImpl(Ctx,arg,
                                                              TypeConfig,
                                                          fullyQualifyTmpltArg);
            desArgs.push_back(arg);
#else
            if (templateArgs[I].getKind() == TemplateArgument::Template) {
               TemplateName tname = templateArgs[I].getAsTemplate();
               // Note: should we not also desugar?
               bool changed = GetFullyQualifiedTemplateName(Ctx, tname);
               if (changed) {
                  desArgs.push_back(TemplateArgument(tname));
                  mightHaveChanged = true;
               } else
                  desArgs.push_back(templateArgs[I]);
               continue;
            }

            if (templateArgs[I].getKind() != TemplateArgument::Type) {
              desArgs.push_back(templateArgs[I]);
              continue;
            }

            QualType SubTy = templateArgs[I].getAsType();
            // Check if the type needs more desugaring and recurse.
            if (isa<TypedefType>(SubTy)
                || isa<TemplateSpecializationType>(SubTy)
                || isa<ElaboratedType>(SubTy)
                || fullyQualifyTmpltArg) {
              mightHaveChanged = true;
              QualType PDQT
                = GetPartiallyDesugaredTypeImpl(Ctx, SubTy, TypeConfig,
                                                /*fullyQualifyType=*/true,
                                                /*fullyQualifyTmpltArg=*/true);
              desArgs.push_back(TemplateArgument(PDQT));
            } else {
              desArgs.push_back(templateArgs[I]);
            }
#endif
          }

          // If desugaring happened allocate new type in the AST.
          if (mightHaveChanged) {
            Qualifiers qualifiers = QT.getLocalQualifiers();
            TemplateName TN(TSTdecl->getSpecializedTemplate());
            QT = Ctx.getTemplateSpecializationType(TN, desArgs,
                                         TSTRecord->getCanonicalTypeInternal());
            QT = Ctx.getQualifiedType(QT, qualifiers);
          }
        }
      }
    }
    // TODO: Find a way to avoid creating new types, if the input is already
    // fully qualified.
    if (prefix) {
      // We intentionally always use ETK_None, we never want
      // the keyword (humm ... what about anonymous types?)
      QT = Ctx.getElaboratedType(ETK_None,prefix,QT);
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    } else if (original_prefix) {
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    }
    return QT;
  }

  QualType Transform::GetPartiallyDesugaredType(const ASTContext& Ctx,
    QualType QT, const Transform::Config& TypeConfig,
    bool fullyQualify/*=true*/)
  {
    return GetPartiallyDesugaredTypeImpl(Ctx,QT,TypeConfig,
                                         /*qualifyType*/fullyQualify,
                                         /*qualifyTmpltArg*/fullyQualify);
  }

  NamespaceDecl* Lookup::Namespace(Sema* S, const char* Name,
                                   const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    LookupResult R(*S, DName, SourceLocation(),
                   Sema::LookupNestedNameSpecifierName);
    R.suppressDiagnostics();
    if (!Within)
      S->LookupName(R, S->TUScope);
    else {
      if (const clang::TagDecl* TD = dyn_cast<clang::TagDecl>(Within)) {
        if (!TD->getDefinition()) {
          // No definition, no lookup result.
          return 0;
        }
      }
      S->LookupQualifiedName(R, const_cast<DeclContext*>(Within));
    }

    if (R.empty())
      return 0;

    R.resolveKind();

    return dyn_cast<NamespaceDecl>(R.getFoundDecl());
  }

  NamedDecl* Lookup::Named(Sema* S, llvm::StringRef Name,
                           const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    return Lookup::Named(S, DName, Within);
  }

  NamedDecl* Lookup::Named(Sema* S, const char* Name,
                           const DeclContext* Within) {
    return Lookup::Named(S, llvm::StringRef(Name), Within);
  }

  NamedDecl* Lookup::Named(Sema* S, const clang::DeclarationName& Name,
                           const DeclContext* Within) {
    LookupResult R(*S, Name, SourceLocation(), Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);
    Lookup::Named(S, R, Within);
    return LookupResult2Decl<clang::NamedDecl>(R);
  }

  TagDecl* Lookup::Tag(Sema* S, llvm::StringRef Name,
                       const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    return Lookup::Tag(S, DName, Within);
  }

  TagDecl* Lookup::Tag(Sema* S, const char* Name,
                       const DeclContext* Within) {
    return Lookup::Tag(S, llvm::StringRef(Name), Within);
  }

  TagDecl* Lookup::Tag(Sema* S, const clang::DeclarationName& Name,
                       const DeclContext* Within) {
    LookupResult R(*S, Name, SourceLocation(), Sema::LookupTagName,
                   Sema::ForRedeclaration);
    Lookup::Named(S, R, Within);
    return LookupResult2Decl<clang::TagDecl>(R);
  }

  void Lookup::Named(Sema* S, LookupResult& R, const DeclContext* Within) {
    R.suppressDiagnostics();
    if (!Within)
      S->LookupName(R, S->TUScope);
    else {
      const DeclContext* primaryWithin = nullptr;
      if (const clang::TagDecl *TD = dyn_cast<clang::TagDecl>(Within)) {
        primaryWithin = dyn_cast_or_null<DeclContext>(TD->getDefinition());
      } else {
        primaryWithin = Within->getPrimaryContext();
      }
      if (!primaryWithin) {
        // No definition, no lookup result.
        return;
      }
      S->LookupQualifiedName(R, const_cast<DeclContext*>(primaryWithin));
    }
  }

  static NestedNameSpecifier*
  CreateNestedNameSpecifierForScopeOf(const ASTContext& Ctx,
                                      const Decl *decl,
                                      bool FullyQualified)
  {
    // Create a nested name specifier for the declaring context of the type.

    assert(decl);

    const NamedDecl* outer
      = llvm::dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
    const NamespaceDecl* outer_ns
      = llvm::dyn_cast_or_null<NamespaceDecl>(decl->getDeclContext());
    if (outer && !(outer_ns && outer_ns->isAnonymousNamespace())) {

      if (const CXXRecordDecl *cxxdecl
          = llvm::dyn_cast<CXXRecordDecl>(decl->getDeclContext())) {

        if (ClassTemplateDecl *clTempl = cxxdecl->getDescribedClassTemplate()) {
          // We are in the case of a type(def) that was declared in a
          // class template but is *not* type dependent.  In clang, it gets
          // attached to the class template declaration rather than any
          // specific class template instantiation.   This result in 'odd'
          // fully qualified typename:
          //    vector<_Tp,_Alloc>::size_type
          // Make the situation is 'useable' but looking a bit odd by
          // picking a random instance as the declaring context.
          // FIXME: We should not use the iterators here to check if we are in
          // a template specialization. clTempl != cxxdecl already tell us that
          // is the case. It seems that we rely on a side-effect from triggering
          // deserializations to support 'some' use-case. See ROOT-9709.
          if (clTempl->spec_begin() != clTempl->spec_end()) {
            decl = *(clTempl->spec_begin());
            outer  = llvm::dyn_cast<NamedDecl>(decl);
            outer_ns = llvm::dyn_cast<NamespaceDecl>(decl);
          }
        }
      }

      if (outer_ns) {
        return TypeName::CreateNestedNameSpecifier(Ctx,outer_ns);
      } else if (const TagDecl* TD = llvm::dyn_cast<TagDecl>(outer)) {
        return TypeName::CreateNestedNameSpecifier(Ctx, TD, FullyQualified);
      }
    }
    return 0;
  }

  static NestedNameSpecifier*
  CreateNestedNameSpecifierForScopeOf(const ASTContext& Ctx,
                                      const Type *TypePtr,
                                      bool FullyQualified)
  {
    // Create a nested name specifier for the declaring context of the type.

    if (!TypePtr)
      return 0;

    Decl *decl = 0;
    if (const TypedefType* typedeftype = llvm::dyn_cast<TypedefType>(TypePtr)) {
      decl = typedeftype->getDecl();
    } else {
      // There are probably other cases ...
      if (const TagType* tagdecltype = llvm::dyn_cast_or_null<TagType>(TypePtr))
        decl = tagdecltype->getDecl();
      else
        decl = TypePtr->getAsCXXRecordDecl();
    }

    if (!decl)
      return 0;

    return CreateNestedNameSpecifierForScopeOf(Ctx, decl, FullyQualified);
  }

  NestedNameSpecifier*
  TypeName::CreateNestedNameSpecifier(const ASTContext& Ctx,
                                      const NamespaceDecl* Namesp) {
    while (Namesp && Namesp->isInline()) {
      // Ignore inline namespace;
      Namesp = dyn_cast_or_null<NamespaceDecl>(Namesp->getDeclContext());
    }
    if (!Namesp) return 0;

    bool FullyQualified = true; // doesn't matter, DeclContexts are namespaces
    return NestedNameSpecifier::Create(Ctx, CreateOuterNNS(Ctx, Namesp,
                                                           FullyQualified),
                                       Namesp);
  }

  NestedNameSpecifier*
  TypeName::CreateNestedNameSpecifier(const ASTContext& Ctx,
                                      const TypedefNameDecl* TD,
                                      bool FullyQualify) {
    return NestedNameSpecifier::Create(Ctx, CreateOuterNNS(Ctx, TD,
                                                           FullyQualify),
                                       true /*Template*/,
                                       TD->getTypeForDecl());
  }

  NestedNameSpecifier*
  TypeName::CreateNestedNameSpecifier(const ASTContext& Ctx,
                                      const TagDecl *TD, bool FullyQualify) {
    const Type* Ty
      = Ctx.getTypeDeclType(TD).getTypePtr();
    if (FullyQualify)
      Ty = GetFullyQualifiedLocalType(Ctx, Ty);
    return NestedNameSpecifier::Create(Ctx,
                                       CreateOuterNNS(Ctx, TD, FullyQualify),
                                       false /* template keyword wanted */,
                                       Ty);
  }

  QualType
  TypeName::GetFullyQualifiedType(QualType QT, const ASTContext& Ctx) {
    // Return the fully qualified type, if we need to recurse through any
    // template parameter, this needs to be merged somehow with
    // GetPartialDesugaredType.

    // In case of myType* we need to strip the pointer first, fully qualifiy
    // and attach the pointer once again.
    if (llvm::isa<PointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();
      QT = GetFullyQualifiedType(QT->getPointeeType(), Ctx);
      QT = Ctx.getPointerType(QT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // In case of myType& we need to strip the pointer first, fully qualifiy
    // and attach the pointer once again.
    if (llvm::isa<ReferenceType>(QT.getTypePtr())) {
      // Get the qualifiers.
      bool isLValueRefTy = llvm::isa<LValueReferenceType>(QT.getTypePtr());
      Qualifiers quals = QT.getQualifiers();
      QT = GetFullyQualifiedType(QT->getPointeeType(), Ctx);
      // Add the r- or l-value reference type back to the desugared one.
      if (isLValueRefTy)
        QT = Ctx.getLValueReferenceType(QT);
      else
        QT = Ctx.getRValueReferenceType(QT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // Strip deduced types.
    if (const AutoType* AutoTy = dyn_cast<AutoType>(QT.getTypePtr())) {
      if (!AutoTy->getDeducedType().isNull())
        return GetFullyQualifiedType(AutoTy->getDeducedType(), Ctx);
    }

    // Remove the part of the type related to the type being a template
    // parameter (we won't report it as part of the 'type name' and it is
    // actually make the code below to be more complex (to handle those)
    while (isa<SubstTemplateTypeParmType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();

      QT = dyn_cast<SubstTemplateTypeParmType>(QT.getTypePtr())->desugar();

      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
    }

    NestedNameSpecifier* prefix = 0;
    Qualifiers prefix_qualifiers;
    if (const ElaboratedType* etype_input
        = llvm::dyn_cast<ElaboratedType>(QT.getTypePtr())) {
      // Intentionally, we do not care about the other compononent of
      // the elaborated type (the keyword) as part of the partial
      // desugaring (and/or name normalization) is to remove it.
      prefix = etype_input->getQualifier();
      if (prefix) {
        const NamespaceDecl *ns = prefix->getAsNamespace();
        if (prefix != NestedNameSpecifier::GlobalSpecifier(Ctx)
            && !(ns && ns->isAnonymousNamespace())) {
          prefix_qualifiers = QT.getLocalQualifiers();
          prefix = GetFullyQualifiedNameSpecifier(Ctx, prefix);
          QT = QualType(etype_input->getNamedType().getTypePtr(),0);
        } else {
          prefix = 0;
        }
      }
    } else {

      // Create a nested name specifier if needed (i.e. if the decl context
      // is not the global scope.
      prefix = CreateNestedNameSpecifierForScopeOf(Ctx,QT.getTypePtr(),
                                                   true /*FullyQualified*/);

      // move the qualifiers on the outer type (avoid 'std::const string'!)
      if (prefix) {
        prefix_qualifiers = QT.getLocalQualifiers();
        QT = QualType(QT.getTypePtr(),0);
      }
    }

    // In case of template specializations iterate over the arguments and
    // fully qualify them as well.
    if(llvm::isa<const TemplateSpecializationType>(QT.getTypePtr())) {

      Qualifiers qualifiers = QT.getLocalQualifiers();
      const Type *TypePtr = GetFullyQualifiedLocalType(Ctx,QT.getTypePtr());
      QT = Ctx.getQualifiedType(TypePtr, qualifiers);

    } else if (llvm::isa<const RecordType>(QT.getTypePtr())) {
      // We are asked to fully qualify and we have a Record Type,
      // which can point to a template instantiation with no sugar in any of
      // its template argument, however we still need to fully qualify them.

      Qualifiers qualifiers = QT.getLocalQualifiers();
      const Type *TypePtr = GetFullyQualifiedLocalType(Ctx,QT.getTypePtr());
      QT = Ctx.getQualifiedType(TypePtr, qualifiers);

    }
    if (prefix) {
      // We intentionally always use ETK_None, we never want
      // the keyword (humm ... what about anonymous types?)
      QT = Ctx.getElaboratedType(ETK_None,prefix,QT);
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    }
    return QT;
  }

  std::string TypeName::GetFullyQualifiedName(QualType QT,
                                              const ASTContext &Ctx) {
    QualType FQQT = GetFullyQualifiedType(QT, Ctx);
    PrintingPolicy Policy(Ctx.getPrintingPolicy());
    Policy.SuppressScope = false;
    Policy.AnonymousTagLocations = false;
    return FQQT.getAsString(Policy);
  }

} // end namespace utils
} // end namespace cling
