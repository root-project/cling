//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ClingPragmas.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Parse/Parser.h"

using namespace cling;
using namespace clang;

namespace {
  class PHLoad: public PragmaHandler {
    Interpreter& m_Interp;

  public:
    PHLoad(Interpreter& interp):
      PragmaHandler("load"), m_Interp(interp) {}

    void HandlePragma(Preprocessor &PP,
                      PragmaIntroducerKind Introducer,
                      Token &FirstToken) override {
      // TODO: use Diagnostics!

      struct SkipToEOD_t {
        Preprocessor& m_PP;
        SkipToEOD_t(Preprocessor& PP): m_PP(PP) {}
        ~SkipToEOD_t() { m_PP.DiscardUntilEndOfDirective(); }
      } SkipToEOD(PP);

      Token Tok;
      PP.Lex(Tok);
      if (Tok.isNot(tok::l_paren)) {
        llvm::errs() << "cling::PHLoad: expect '(' after #pragma cling load!\n";
        return;
      }
      std::string FileName;
      if (!PP.LexStringLiteral(Tok, FileName, "pragma cling load",
                               false /*allowMacroExpansion*/)) {
        // already diagnosed.
        return;
      }
      Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
      clang::Parser& P = m_Interp.getParser();
      Parser::ParserCurTokRestoreRAII savedCurToken(P);
      // After we have saved the token reset the current one to something which
      // is safe (semi colon usually means empty decl)
      Token& CurTok = const_cast<Token&>(P.getCurToken());
      CurTok.setKind(tok::semi);

      // We can't PushDeclContext, because we go up and the routine that pops
      // the DeclContext assumes that we drill down always.
      // We have to be on the global context. At that point we are in a
      // wrapper function so the parent context must be the global.
      TranslationUnitDecl* TU
        = m_Interp.getCI()->getASTContext().getTranslationUnitDecl();
      Sema::ContextAndScopeRAII pushedDCAndS(m_Interp.getSema(),
                                             TU, m_Interp.getSema().TUScope);

      Interpreter::PushTransactionRAII pushedT(&m_Interp);
      m_Interp.loadFile(FileName, true /*allowSharedLib*/);
    }
  };
}

void cling::addClingPragmas(Interpreter& interp) {
  Preprocessor& PP = interp.getCI()->getPreprocessor();
  // PragmaNamespace / PP takes ownership of sub-handlers.
  PP.AddPragmaHandler("cling", new PHLoad(interp));
}
