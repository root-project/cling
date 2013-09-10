//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: cb7241880ebcbba87b2ae16476c2812afd7ff571 $
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_AST_NULL_DEREF_PROTECTION_H
#define CLING_AST_NULL_DEREF_PROTECTION_H

#include "TransactionTransformer.h"
#include "clang/AST/Decl.h"

#include <bitset>
#include <map>

namespace clang {
  class CallExpr;
  class Expr;
  class FunctionDecl;
  class Sema;
  class Stmt;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {
  typedef std::map<clang::FunctionDecl*, std::bitset<32> > decl_map_t;

  class ASTNullDerefProtection : public TransactionTransformer {

    private:
      std::map<clang::FunctionDecl*, std::bitset<32> > m_NonNullArgIndexs;
      bool isDeclCandidate(clang::FunctionDecl* FDecl);

    public:
      ///\ brief Constructs the NullDeref AST Transformer.
      ///
      ///\param[in] S - The semantic analysis object.
      ///
      ASTNullDerefProtection(clang::Sema* S);

      virtual ~ASTNullDerefProtection();
      clang::Expr* InsertThrow(clang::SourceLocation* Loc,
                               clang::Expr* Arg);
      clang::Stmt* SynthesizeCheck(clang::SourceLocation Loc,
                                   clang::Expr* Arg);
      virtual void Transform();
  };

} // namespace cling

#endif // CLING_AST_NULL_DEREF_PROTECTION_H
