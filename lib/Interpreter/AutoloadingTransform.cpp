#include "AutoloadingTransform.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  AutoloadingTransform::AutoloadingTransform(clang::Sema* S)
    : TransactionTransformer(S) {
  }

  AutoloadingTransform::~AutoloadingTransform()
  {}

  void AutoloadingTransform::Transform() {
    const Transaction* T = getTransaction();
    for (Transaction::const_iterator I = T->decls_begin(), E = T->decls_end();
         I != E; ++I) {
      Transaction::DelayCallInfo DCI = *I;
      for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
             JE = DCI.m_DGR.end(); J != JE; ++J) {
        if ( (*J)->hasAttr<AnnotateAttr>() /*FIXME: && CorrectCallbackLoaded() how ? */  )
          clang::Decl::castToDeclContext(*J)->setHasExternalLexicalStorage();

      }
    }
  }
} // end namespace cling
