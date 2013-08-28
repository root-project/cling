//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/DeclTemplate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <stdio.h>
using namespace clang;

namespace cling {
namespace utils {

  static
  QualType GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx, 
                                         QualType QT, 
                               const llvm::SmallSet<const Type*,4>& TypesToSkip,
                                         bool fullyQualifyType,
                                         bool fullyQualifyTmpltArg);

  static
  NestedNameSpecifier* GetPartiallyDesugaredNNS(const ASTContext& Ctx,
                                                NestedNameSpecifier* scope, 
                             const llvm::SmallSet<const Type*, 4>& TypesToSkip);

  bool Analyze::IsWrapper(const NamedDecl* ND) {
    if (!ND)
      return false;

    return StringRef(ND->getNameAsString())
      .startswith(Synthesize::UniquePrefix);
  }

  bool Analyze::IsCFWrapper(const NamedDecl* ND) {
    if (!ND) {
      return false;
    }
    bool ret = StringRef(ND->getNameAsString()).startswith(
      Synthesize::CFUniquePrefix);
    return ret;
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
              Expr* DRE = S->BuildDeclRefExpr(VD, VDTy,VK_LValue, Loc).take();
              assert(DRE && "Cannot be null");
              indexOfLastExpr++;
              newBody.insert(newBody.begin() + indexOfLastExpr, DRE);

              // Attach the new body (note: it does dealloc/alloc of all nodes)
              CS->setStmts(S->getASTContext(), &newBody.front(), newBody.size());
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
  const char* const Synthesize::CFUniquePrefix = "__cf_Un1Qu3";

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, uint64_t Ptr) {
    ASTContext& Ctx = S->getASTContext();
    return CStyleCastPtrExpr(S, Ty, Synthesize::IntegerLiteralExpr(Ctx, Ptr));
  }

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, Expr* E) {
    ASTContext& Ctx = S->getASTContext();
    if (!Ty->isPointerType())
      Ty = Ctx.getPointerType(Ty);

    TypeSourceInfo* TSI = Ctx.getTrivialTypeSourceInfo(Ty, SourceLocation());
    Expr* Result 
      = S->BuildCStyleCastExpr(SourceLocation(), TSI,SourceLocation(),E).take();
    assert(Result && "Cannot create CStyleCastPtrExpr");
    return Result;
  }

  IntegerLiteral* Synthesize::IntegerLiteralExpr(ASTContext& C, uint64_t Ptr) {
    const llvm::APInt Addr(8 * sizeof(void *), Ptr);
    return IntegerLiteral::Create(C, Addr, C.UnsignedLongTy, SourceLocation());
  }

  static
  NestedNameSpecifier* CreateNestedNameSpecifier(const ASTContext& Ctx,
                                                 const NamespaceDecl* cl) {
    
    const NamespaceDecl* outer
      = dyn_cast_or_null<NamespaceDecl>(cl->getDeclContext());
    if (outer && outer->getName().size()) {
      NestedNameSpecifier* outerNNS = CreateNestedNameSpecifier(Ctx,outer);
      return NestedNameSpecifier::Create(Ctx,outerNNS,
                        // Newer version of clang do not require this const_cast
                                         const_cast<NamespaceDecl*>(cl));
    } else {
      return NestedNameSpecifier::Create(Ctx, 
                                         0, /* no starting '::'*/
                       // Newer version of clang do not require this const_cast
                                         const_cast<NamespaceDecl*>(cl));
    }
  }
  
  static
  NestedNameSpecifier* CreateNestedNameSpecifier(const ASTContext& Ctx,
                                                 const TagDecl *cl) {
    
    const NamedDecl* outer = dyn_cast_or_null<NamedDecl>(cl->getDeclContext());
    if (outer && outer->getName().size()) {
      NestedNameSpecifier *outerNNS;
      if (cl->getDeclContext()->isNamespace()) {
        outerNNS = CreateNestedNameSpecifier(Ctx,
                                             dyn_cast<NamespaceDecl>(outer));
      } else {
        outerNNS = CreateNestedNameSpecifier(Ctx,
                                             dyn_cast<TagDecl>(outer));
      }
      return NestedNameSpecifier::Create(Ctx,outerNNS,
                                         false /* template keyword wanted */,
                                         Ctx.getTypeDeclType(cl).getTypePtr());
    } else {
      return NestedNameSpecifier::Create(Ctx, 
                                         0, /* no starting '::'*/
                                         false /* template keyword wanted */,
                                         Ctx.getTypeDeclType(cl).getTypePtr());        
    }
  }
  
  static
  NestedNameSpecifier* GetFullyQualifiedNameSpecifier(const ASTContext& Ctx,
                                                  NestedNameSpecifier* scope) {
    // Return a fully qualified version of this name specifier
    NestedNameSpecifier *outer_nns = scope;
    while( outer_nns->getPrefix()
          && outer_nns->getKind() != NestedNameSpecifier::Global) {
      outer_nns = outer_nns->getPrefix();
    }
    Decl* decl = 0;
    if (outer_nns->getKind() == NestedNameSpecifier::Global) {
      // leave decl to 0.
    } else if (const Type *type = outer_nns->getAsType()) {
      // Find decl context.
      const TypedefType* typedeftype = dyn_cast_or_null<TypedefType>(type);
      if (typedeftype) {
        decl = typedeftype->getDecl();
      } else {
        // There are probably other cases ...
        const TagType* tagdecltype = dyn_cast_or_null<TagType>(type);
        if (tagdecltype) {
          decl = tagdecltype->getDecl();
        } else {
          decl = type->getAsCXXRecordDecl();
        }
      }
    } else if ( (decl = outer_nns->getAsNamespace()) ) {
      // Found decl.
    } else if ( (decl = outer_nns->getAsNamespaceAlias()) ) {
      // Found decl.
    }
    
    bool needCreate = false;
    if (decl == 0) {
      // We have the global namespace in there, we don't want it.
      needCreate = true;
    } else {
      NamedDecl* outer = dyn_cast<NamedDecl>(decl->getDeclContext());
      NamespaceDecl* outer_ns = dyn_cast<NamespaceDecl>(decl->getDeclContext());
      if (outer
          && !(outer_ns && outer_ns->isAnonymousNamespace())
          && outer->getName().size() )
      {
        needCreate = true;
      }
    }
    if (needCreate) {
      if (NamespaceDecl *ns = scope->getAsNamespace()) {
        return CreateNestedNameSpecifier(Ctx,ns);
      } else if (NamespaceAliasDecl *alias = scope->getAsNamespaceAlias())
      {
        return CreateNestedNameSpecifier(Ctx,
                                     alias->getNamespace()->getCanonicalDecl());

      } else {
        // We should only create the nested name specifier
        // if the outer scope is really a TagDecl.
        // It could also be a CXXMethod for example.
        const Type *type = scope->getAsType();
        const TypedefType* typedeftype = dyn_cast_or_null<TypedefType>(type);
        Decl *idecl;
        if (typedeftype) {
          idecl = typedeftype->getDecl();
        } else {
          // There are probably other cases ...
          const TagType* tagdecltype = dyn_cast_or_null<TagType>(type);
          if (tagdecltype) {
            idecl = tagdecltype->getDecl();
          } else {
            idecl = type->getAsCXXRecordDecl();
          }
        }
        TagDecl *tdecl = dyn_cast<TagDecl>(idecl);
        if (tdecl) {
          return CreateNestedNameSpecifier(Ctx,tdecl);
        }
      }
    }
    // It was fine.
    return scope;
  }
  
  static
  NestedNameSpecifier* SelectPrefix(const ASTContext& Ctx,
                                    const DeclContext *declContext,
                                    NestedNameSpecifier *original_prefix,
                             const llvm::SmallSet<const Type*,4>& TypesToSkip) {
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
            if (old_ns) {
              old_ns = old_ns->getCanonicalDecl();
            }
            else if (NamespaceAliasDecl *alias =
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
            prefix = CreateNestedNameSpecifier(Ctx,
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
            prefix = GetPartiallyDesugaredNNS(Ctx,original_prefix,TypesToSkip);
          } else {
            const TagDecl *tdecl = dyn_cast<TagDecl>(declContext);
            if (tdecl) {
              prefix = CreateNestedNameSpecifier(Ctx,tdecl);
            }
          }
        } else {
          // We should only create the nested name specifier
          // if the outer scope is really a TagDecl.
          // It could also be a CXXMethod for example.
          const TagDecl *tdecl = dyn_cast<TagDecl>(declContext);
          if (tdecl) {
            prefix = CreateNestedNameSpecifier(Ctx,tdecl);
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
                             const llvm::SmallSet<const Type*,4>& TypesToSkip) {
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
          prefix = GetPartiallyDesugaredNNS(Ctx,original_prefix,TypesToSkip);
        } else {
          prefix = GetPartiallyDesugaredNNS(Ctx,prefix,TypesToSkip);
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
                            const llvm::SmallSet<const Type*, 4>& TypesToSkip){
    // Desugar the scope qualifier if needed.

    if (const Type* scope_type = scope->getAsType()) {
      
      // this is not a namespace, so we might need to desugar
      QualType desugared = GetPartiallyDesugaredTypeImpl(Ctx,
                                                         QualType(scope_type,0),
                                                         TypesToSkip,
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
          outer_scope = SelectPrefix(Ctx,etype,outer_scope,TypesToSkip);
        } else {
          outer_scope = GetPartiallyDesugaredNNS(Ctx,etype->getQualifier(),
                                                 TypesToSkip);
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
                                       outer_scope,TypesToSkip);
          } else {
            outer_scope = 0;
          }
        } else if (outer_scope) {
          outer_scope = GetPartiallyDesugaredNNS(Ctx, outer_scope, TypesToSkip);
        }
      }
      return NestedNameSpecifier::Create(Ctx,outer_scope,
                                         false /* template keyword wanted */,
                                         desugared.getTypePtr());
    } else {
      return  GetFullyQualifiedNameSpecifier(Ctx,scope);
    }
  }

  static bool IsStdDetails(const TagType *tagTy)
  {
    // Return true if the TagType is a 'details' of the std implementation.
    // (For now it means declared in std and __gnu_cxx
    
    const TagDecl *decl = tagTy->getDecl();
    assert(decl);
    const NamedDecl *outer =dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
    while (outer && outer->getName().size() ) {
      if (outer->getName().compare("std") == 0 ||
          outer->getName().compare("__gnu_cxx") == 0) {
        return true;
      }
      outer = dyn_cast_or_null<NamedDecl>(outer->getDeclContext());
    }
    return false;
  }
  
  static bool ShouldKeepTypedef(QualType QT,
                           const llvm::SmallSet<const Type*, 4>& TypesToSkip)
  {
    // Return true, if we should keep this typedef rather than desugaring it.

    if ( 0 != TypesToSkip.count(QT.getTypePtr()) ) 
      return true;
     
    const TypedefType* typedeftype = 
      dyn_cast_or_null<TypedefType>(QT.getTypePtr());
    const TypedefNameDecl* decl = typedeftype ? typedeftype->getDecl() : 0;
    if (decl) {
      const NamedDecl* outer 
        = dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
      // We want to keep the typedef that are defined within std and
      // are pointing to something also declared in std (usually an
      // implementation details like std::basic_string or __gnu_cxx::iterator.
      
      while ( outer && outer->getName().size() ) {
        // NOTE: Net is being cast too widely, replace by a lookup.
        // or by using Sema::getStdNamespace
        if (outer->getName().compare("std") == 0) {
          // And now let's check that the target is also within std.
          const Type *underlyingType = decl->getUnderlyingType().getSplitDesugaredType().Ty;
          const ElaboratedType *elTy = dyn_cast<ElaboratedType>(underlyingType);
          if (elTy) {
            underlyingType = elTy->getNamedType().getTypePtr();
          }
          const TagType *tagTy = underlyingType->getAs<TagType>();
          if (tagTy) {
            bool details = IsStdDetails(tagTy);
            if (details) return true;
          }
        }
        outer = dyn_cast_or_null<NamedDecl>(outer->getDeclContext());
      }
    }
    return false;
  }

  bool SingleStepPartiallyDesugarTypeImpl(QualType& QT)
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
        return false;
        //const TemplateSpecializationType* Ty =
        //  llvm::cast<TemplateSpecializationType>(QTy);
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
   
  static QualType GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx, 
    QualType QT, const llvm::SmallSet<const Type*,4>& TypesToSkip,
    bool fullyQualifyType,
    bool fullyQualifyTmpltArg)                                                
  {
    // If there are no constraints, then use the standard desugaring.
    if (!TypesToSkip.size() && !fullyQualifyType && !fullyQualifyTmpltArg)
      return QT.getDesugaredType(Ctx);

    // In case of Int_t* we need to strip the pointer first, desugar and attach
    // the pointer once again.
    if (isa<PointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();      
      QT = GetPartiallyDesugaredTypeImpl(Ctx, QT->getPointeeType(), TypesToSkip, 
                                         fullyQualifyType,fullyQualifyTmpltArg);
      QT = Ctx.getPointerType(QT);
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
    // the pointer once again.
    if (isa<ReferenceType>(QT.getTypePtr())) {
      // Get the qualifiers.
      bool isLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
      Qualifiers quals = QT.getQualifiers();
      QT = GetPartiallyDesugaredTypeImpl(Ctx, QT->getPointeeType(), TypesToSkip, 
                                         fullyQualifyType,fullyQualifyTmpltArg);
      // Add the r- or l-value reference type back to the desugared one.
      if (isLValueRefTy)
        QT = Ctx.getLValueReferenceType(QT);
      else
        QT = Ctx.getRValueReferenceType(QT);
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
      if (llvm::isa<TypedefType>(QT.getTypePtr()) &&
          ShouldKeepTypedef(QT, TypesToSkip)) {
        if (!fullyQualifyType && !fullyQualifyTmpltArg) {
          return QT;
        }
        // We might have stripped the namespace/scope part,
        // so we must go on to add it back.
        break;
      }
      bool wasDesugared = Transform::SingleStepPartiallyDesugarType(QT,Ctx);
      if (!wasDesugared) {
        // No more work to do, stop now.
        break;
      }
    }

    // If we have a reference or pointer we still need to
    // desugar what they point to.
    if (isa<PointerType>(QT.getTypePtr()) ||
        isa<ReferenceType>(QT.getTypePtr()) ) {
      return GetPartiallyDesugaredTypeImpl(Ctx, QT, TypesToSkip, 
                                           fullyQualifyType,
                                           fullyQualifyTmpltArg);
    }

    NestedNameSpecifier* prefix = 0;
    const ElaboratedType* etype 
      = dyn_cast<ElaboratedType>(QT.getTypePtr());
    if (etype) {

      prefix = SelectPrefix(Ctx,etype,original_prefix,TypesToSkip);
      
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
        const TagType* tagdecltype = 
           dyn_cast_or_null<TagType>(QT.getTypePtr());
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
            && outer->getName().size() ) {
          if (original_prefix) {
            const Type *oldtype = original_prefix->getAsType();
            if (oldtype) {
              if (oldtype->getAsCXXRecordDecl() == outer) {
                // Same type, use the original spelling
                prefix = GetPartiallyDesugaredNNS(Ctx,original_prefix,TypesToSkip);
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
            // move the qualifiers on the outer type (avoid 'std::const string'!)
            prefix_qualifiers = QT.getLocalQualifiers();
            QT = QualType(QT.getTypePtr(),0);
          }
          if (outer) {
            if (decl->getDeclContext()->isNamespace()) {
              prefix = CreateNestedNameSpecifier(Ctx,
                                                dyn_cast<NamespaceDecl>(outer));
            } else {
              // We should only create the nested name specifier
              // if the outer scope is really a TagDecl.
              // It could also be a CXXMethod for example.
              TagDecl *tdecl = dyn_cast<TagDecl>(outer);
              if (tdecl) {
                prefix = CreateNestedNameSpecifier(Ctx,tdecl);
                prefix = GetPartiallyDesugaredNNS(Ctx,prefix,TypesToSkip);
              }
            }
          }
        }
      }
    }

    // In case of template specializations iterate over the arguments and 
    // desugar them as well.
    if(const TemplateSpecializationType* TST 
       = dyn_cast<const TemplateSpecializationType>(QT.getTypePtr())) {
     
      bool mightHaveChanged = false;
      llvm::SmallVector<TemplateArgument, 4> desArgs;
      for(TemplateSpecializationType::iterator I = TST->begin(), E = TST->end();
          I != E; ++I) {
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
          mightHaveChanged = true;
          desArgs.push_back(TemplateArgument(GetPartiallyDesugaredTypeImpl(Ctx,
                                                                       SubTy,
                                                                     TypesToSkip, 
                                                                fullyQualifyType,
                                                         fullyQualifyTmpltArg)));
       } else 
          desArgs.push_back(*I);
      }
      
      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
        Qualifiers qualifiers = QT.getLocalQualifiers();
        QT = Ctx.getTemplateSpecializationType(TST->getTemplateName(), 
                                               desArgs.data(),
                                               desArgs.size(),
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
              desArgs.push_back(TemplateArgument(
                                             GetPartiallyDesugaredTypeImpl(Ctx,
                                                                         SubTy,
                                                                   TypesToSkip,
                                                     /*fullyQualifyType=*/true,
                                              /*fullyQualifyTmpltArg=*/true)));
            } else
              desArgs.push_back(templateArgs[I]);
          }
       
          // If desugaring happened allocate new type in the AST.
          if (mightHaveChanged) {
            Qualifiers qualifiers = QT.getLocalQualifiers();
            QT = Ctx.getTemplateSpecializationType(TemplateName(TSTdecl->getSpecializedTemplate()),
                                                   desArgs.data(),
                                                   desArgs.size(),
                                                   TSTRecord->getCanonicalTypeInternal());
            QT = Ctx.getQualifiedType(QT, qualifiers);
          }
        }
      }
    }
    if (prefix) {
      // We intentionally always use ETK_None, we never want
      // the keyword (humm ... what about anonymous types?)
      QT = Ctx.getElaboratedType(ETK_None,prefix,QT);
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    }
    return QT;   
  }

  QualType Transform::GetPartiallyDesugaredType(const ASTContext& Ctx, 
    QualType QT, const llvm::SmallSet<const Type*,4>& TypesToSkip,
    bool fullyQualify/*=true*/)
  {
    return GetPartiallyDesugaredTypeImpl(Ctx,QT,TypesToSkip,
                                         /*qualifyType*/fullyQualify,
                                         /*qualifyTmpltArg*/fullyQualify);
  }
        
  NamespaceDecl* Lookup::Namespace(Sema* S, const char* Name,
                                   const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    LookupResult R(*S, DName, SourceLocation(),
                   Sema::LookupNestedNameSpecifierName);
    if (!Within)
      S->LookupName(R, S->TUScope);
    else
      S->LookupQualifiedName(R, const_cast<DeclContext*>(Within));

    if (R.empty())
      return 0;

    R.resolveKind();

    return dyn_cast<NamespaceDecl>(R.getFoundDecl());
  }

  NamedDecl* Lookup::Named(Sema* S, const char* Name,
                           const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    return Lookup::Named(S, DName, Within);
  }

  NamedDecl* Lookup::Named(Sema* S, const DeclarationName& Name, 
                           const DeclContext* Within) {
    LookupResult R(*S, Name, SourceLocation(), Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);
    if (!Within)
      S->LookupName(R, S->TUScope);
    else
      S->LookupQualifiedName(R, const_cast<DeclContext*>(Within));

    if (R.empty())
      return 0;

    R.resolveKind();

    return R.getFoundDecl();

  }
} // end namespace utils
} // end namespace cling
