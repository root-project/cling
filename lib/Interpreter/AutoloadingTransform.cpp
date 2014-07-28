#include "clang/Sema/Sema.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/AST.h"

#include "AutoloadingTransform.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/DynamicLibraryManager.h"

using namespace clang;

namespace cling {
  AutoloadingTransform::AutoloadingTransform(clang::Sema* S, Interpreter*)
    : TransactionTransformer(S) {
  }

  AutoloadingTransform::~AutoloadingTransform()
  {}

  void AutoloadingTransform::Transform() {
    const Transaction* T = getTransaction();
    for (Transaction::const_iterator I = T->decls_begin(), E = T->decls_end();
         I != E; ++I) {
      Transaction::DelayCallInfo DCI = *I;
      std::vector<clang::Decl*> decls;
      for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
             JE = DCI.m_DGR.end(); J != JE; ++J) {
        if (EnumDecl* ED = dyn_cast<EnumDecl>(*J))
          if (ED->hasAttr<AnnotateAttr>() && ED->isFixed()) {
            struct EnumDeclDerived: public EnumDecl {
              static void setFixed(EnumDecl* ED, bool value = true) {
                ((EnumDeclDerived*)ED)->IsFixed = value;
              }
            };
            EnumDeclDerived::setFixed(ED, false);
          }
//FIXME: Enable when safe !
//        if ( (*J)->hasAttr<AnnotateAttr>() /*FIXME: && CorrectCallbackLoaded() how ? */  )
//          clang::Decl::castToDeclContext(*J)->setHasExternalLexicalStorage();
      }
    }
  }
} // end namespace cling
