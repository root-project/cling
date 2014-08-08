//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Manasij Mukherjee  <manasij7479@gmail.com>
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "AutoloadingTransform.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/DeclVisitor.h"

using namespace clang;

namespace cling {

  class DeclFixer : public DeclVisitor<DeclFixer> {
  public:
    void VisitDecl(Decl* D) {
      if (DeclContext* DC = dyn_cast<DeclContext>(D))
        for (auto Child : DC->decls())
          Visit(Child);
    }

    void VisitEnumDecl(EnumDecl* ED) {
      if (ED->isFixed()) {
        StringRef str = ED->getAttr<AnnotateAttr>()->getAnnotation();
        char ch = str.back();
        str.drop_back(2);
        ED->getAttr<AnnotateAttr>()->setAnnotation(ED->getASTContext(), str);
        struct EnumDeclDerived: public EnumDecl {
          static void setFixed(EnumDecl* ED, bool value = true) {
            ((EnumDeclDerived*)ED)->IsFixed = value;
          }
        };

        if (ch != '1')
          EnumDeclDerived::setFixed(ED, false);
      }
    }
  };

  void AutoloadingTransform::Transform() {
    const Transaction* T = getTransaction();
    DeclFixer visitor;
    for (Transaction::const_iterator I = T->decls_begin(), E = T->decls_end();
         I != E; ++I) {
      Transaction::DelayCallInfo DCI = *I;
      for (DeclGroupRef::iterator J = DCI.m_DGR.begin(),
             JE = DCI.m_DGR.end(); J != JE; ++J) {
        if (!(*J)->hasAttr<AnnotateAttr>())
          continue;
        visitor.Visit(*J);
//FIXME: Enable when safe !
//        if ( (*J)->hasAttr<AnnotateAttr>() /*FIXME: && CorrectCallbackLoaded() how ? */  )
//          clang::Decl::castToDeclContext(*J)->setHasExternalLexicalStorage();
      }
    }
  }
} // end namespace cling
