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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;

namespace cling {
namespace utils {

  bool Analyze::IsWrapper(const NamedDecl* ND) {
    if (!ND)
      return false;

    return llvm::StringRef(ND->getNameAsString())
      .startswith(Synthesize::UniquePrefix);
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
      llvm::ArrayRef<Stmt*> Stmts 
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

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, uint64_t Ptr) {
    ASTContext& Ctx = S->getASTContext();
    if (!Ty->isPointerType())
      Ty = Ctx.getPointerType(Ty);
    TypeSourceInfo* TSI = Ctx.CreateTypeSourceInfo(Ty);
    const llvm::APInt Addr(8 * sizeof(void *), Ptr);

    Expr* Result = IntegerLiteral::Create(Ctx, Addr, Ctx.UnsignedLongTy,
                                          SourceLocation());
    Result = S->BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(),
                                         Result).take();
    assert(Result && "Cannot create CStyleCastPtrExpr");
    return Result;
  }

  static
  NestedNameSpecifier* GetPartiallyDesugaredNNS(const ASTContext& Ctx, 
                                                NestedNameSpecifier* scope, 
                            const llvm::SmallSet<const Type*, 4>& TypesToSkip){
    // Desugar the scope qualifier if needed.

    const Type* scope_type = scope->getAsType();
    if (scope_type) {
      // this is not a namespace, so we might need to desugar
      NestedNameSpecifier* outer_scope = scope->getPrefix();
      if (outer_scope) {
        outer_scope = GetPartiallyDesugaredNNS(Ctx, outer_scope, TypesToSkip);
      }

      QualType desugared = 
        Transform::GetPartiallyDesugaredType(Ctx,
                                             QualType(scope_type,0),
                                             TypesToSkip, 
                                             /*fullyQualify=*/false);
      // NOTE: Should check whether the type has changed or not.
      return NestedNameSpecifier::Create(Ctx,outer_scope,
                                         false /* template keyword wanted */,
                                         desugared.getTypePtr());
    }
    return scope;
  }

  static
  NestedNameSpecifier* CreateNestedNameSpecifier(const ASTContext& Ctx,
                                                 NamespaceDecl* cl) {

     NamespaceDecl* outer 
        = llvm::dyn_cast_or_null<NamespaceDecl>(cl->getDeclContext());
     if (outer && outer->getName().size()) {
        NestedNameSpecifier* outerNNS = CreateNestedNameSpecifier(Ctx,outer);
        return NestedNameSpecifier::Create(Ctx,outerNNS,
                                           cl);
     } else {
        return NestedNameSpecifier::Create(Ctx, 
                                           0, /* no starting '::'*/
                                           cl);        
     }
  }

  static
  NestedNameSpecifier* CreateNestedNameSpecifier(const ASTContext& Ctx,
                                                 TagDecl *cl) {

    NamedDecl* outer = llvm::dyn_cast_or_null<NamedDecl>(cl->getDeclContext());
      if (outer && outer->getName().size()) {
        NestedNameSpecifier *outerNNS;
        if (cl->getDeclContext()->isNamespace()) {
          outerNNS = CreateNestedNameSpecifier(Ctx,
                                       llvm::dyn_cast<NamespaceDecl>(outer));
        } else {
          outerNNS = CreateNestedNameSpecifier(Ctx,
                                          llvm::dyn_cast<TagDecl>(outer));
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

  static bool ShouldKeepTypedef(QualType QT, 
                           const llvm::SmallSet<const Type*, 4>& TypesToSkip)
  {
    // Return true, if we should keep this typedef rather than desugaring it.

    if ( 0 != TypesToSkip.count(QT.getTypePtr()) ) 
      return true;
     
    const TypedefType* typedeftype = 
       llvm::dyn_cast_or_null<clang::TypedefType>(QT.getTypePtr());
    const TypedefNameDecl* decl = typedeftype ? typedeftype->getDecl() : 0;
    if (decl) {
       const NamedDecl* outer 
          = llvm::dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
       while ( outer && outer->getName().size() ) {
        // NOTE: Net is being cast to widely, replace by a lookup. 
        if (outer->getName().compare("std") == 0) {
          return true;
        }
        outer = llvm::dyn_cast_or_null<NamedDecl>(outer->getDeclContext());
      }
    }
    return false;
  }

  QualType Transform::GetPartiallyDesugaredType(const ASTContext& Ctx, 
                                                QualType QT, 
                               const llvm::SmallSet<const Type*, 4>& TypesToSkip,
                                                bool fullyQualify /*=true*/){
    // If there are no constains - use the standard desugaring.
    if (!TypesToSkip.size() && !fullyQualify)
      return QT.getDesugaredType(Ctx);

    // In case of Int_t* we need to strip the pointer first, desugar and attach
    // the pointer once again.
    if (isa<PointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();      
      QT = GetPartiallyDesugaredType(Ctx, QT->getPointeeType(), TypesToSkip, 
                                     fullyQualify);
      QT = Ctx.getPointerType(QT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // In case of Int_t& we need to strip the pointer first, desugar and attach
    // the pointer once again.
    if (isa<ReferenceType>(QT.getTypePtr())) {
      // Get the qualifiers.
      bool isLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
      Qualifiers quals = QT.getQualifiers();
      QT = GetPartiallyDesugaredType(Ctx, QT->getPointeeType(), TypesToSkip, 
                                     fullyQualify);
      // Add the r- or l- value reference type back to the desugared one
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
    clang::Qualifiers prefix_qualifiers;
    const ElaboratedType* etype_input 
      = llvm::dyn_cast<ElaboratedType>(QT.getTypePtr());
    if (etype_input) {
      // We have to also desugar the prefix. 
      fullyQualify = true;
      original_prefix = etype_input->getQualifier();
      prefix_qualifiers = QT.getLocalQualifiers();
      QT = QualType(etype_input->getNamedType().getTypePtr(),0);
    }

    while(isa<TypedefType>(QT.getTypePtr())) {
      if (!ShouldKeepTypedef(QT,TypesToSkip))
        QT = QT.getSingleStepDesugaredType(Ctx);
      else if (fullyQualify) {
        // We might have stripped the namespace/scope part,
        // se we must go on if fullyQualify is true.
        break;
      } else
        return QT;
    }

    // If we have a reference or pointer we still need to
    // desugar what they point to.
    if (isa<PointerType>(QT.getTypePtr()) || 
        isa<ReferenceType>(QT.getTypePtr()) ) {
      return GetPartiallyDesugaredType(Ctx, QT, TypesToSkip, 
                                        fullyQualify);
    }

    NestedNameSpecifier* prefix = 0;
    const ElaboratedType* etype 
      = llvm::dyn_cast<ElaboratedType>(QT.getTypePtr());
    if (etype) {
      // We have to also desugar the prefix.
       
      prefix = etype->getQualifier();
      if (original_prefix) {
        // We had a scope prefix as input, let see if it is still
        // the same as the scope of the result and if it is, then
        // we use it.
        const clang::Type *newtype = prefix->getAsType();
        if (newtype) {
          // Deal with a class
          const clang::Type *oldtype = original_prefix->getAsType();
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
          else if (NamespaceAliasDecl *alias = prefix->getAsNamespaceAlias())
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
              prefix = original_prefix;
            }
          }
        }
      }
      prefix_qualifiers.addQualifiers(QT.getLocalQualifiers());
      QT = QualType(etype->getNamedType().getTypePtr(),0);
    } else if (fullyQualify) {
      // Let's check whether this type should have been an elaborated type.
      // in which case we want to add it ... but we can't really preserve
      // the typedef in this case ...

      Decl *decl = 0;
      const TypedefType* typedeftype = 
        llvm::dyn_cast_or_null<clang::TypedefType>(QT.getTypePtr());
      if (typedeftype) {
        decl = typedeftype->getDecl();
      } else {
        // There are probably other cases ...
        const TagType* tagdecltype = 
           llvm::dyn_cast_or_null<clang::TagType>(QT.getTypePtr());
        if (tagdecltype) {
          decl = tagdecltype->getDecl();
        } else {
          decl = QT->getAsCXXRecordDecl();
        }
      }
      if (decl) {
        NamedDecl* outer 
           = llvm::dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
        if (outer && outer->getName ().size()) {
          if (original_prefix) {
            const clang::Type *oldtype = original_prefix->getAsType();
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
              const NamespaceDecl *new_ns = llvm::dyn_cast<NamespaceDecl>(outer);
              if (new_ns) new_ns = new_ns->getCanonicalDecl();
              if (old_ns == new_ns) {
                // This is the same namespace, use the original prefix
                // as a starting point.
                prefix = original_prefix;
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
                                          llvm::dyn_cast<NamespaceDecl>(outer));
            } else {
              prefix = CreateNestedNameSpecifier(Ctx,
                                          llvm::dyn_cast<TagDecl>(outer));
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
        if (I->getKind() != clang::TemplateArgument::Type) {
          desArgs.push_back(*I);
          continue;
        }

        QualType SubTy = I->getAsType();
        // Check if the type needs more desugaring and recurse.
        if (isa<TypedefType>(SubTy) 
            || isa<TemplateSpecializationType>(SubTy)
            || isa<ElaboratedType>(SubTy)
            || fullyQualify) {
          mightHaveChanged = true;
          desArgs.push_back(TemplateArgument(GetPartiallyDesugaredType(Ctx,
                                                                       SubTy,
                                                                     TypesToSkip,
                                                                 fullyQualify)));
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
    }
    if (prefix) {
      QT = Ctx.getElaboratedType(ETK_None,prefix,QT);
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    }
    return QT;   
  }

  NamespaceDecl* Lookup::Namespace(Sema* S, const char* Name,
                                   DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    LookupResult R(*S, DName, SourceLocation(),
                   Sema::LookupNestedNameSpecifierName);
    if (!Within)
      S->LookupName(R, S->TUScope);
    else
      S->LookupQualifiedName(R, Within);

    if (R.empty())
      return 0;

    R.resolveKind();

    return dyn_cast<NamespaceDecl>(R.getFoundDecl());
  }

  NamedDecl* Lookup::Named(Sema* S, const char* Name, DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    return Lookup::Named(S, DName, Within);
  }

  NamedDecl* Lookup::Named(Sema* S, const DeclarationName& Name, 
                           DeclContext* Within) {
    LookupResult R(*S, Name, SourceLocation(), Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);
    if (!Within)
      S->LookupName(R, S->TUScope);
    else
      S->LookupQualifiedName(R, Within);

    if (R.empty())
      return 0;

    R.resolveKind();

    return R.getFoundDecl();

  }
} // end namespace utils
} // end namespace cling
