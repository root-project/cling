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
  static void replaceEnvVars(std::string &Path) {
    std::size_t bpos = Path.find("$");
    while (bpos != std::string::npos) {
      std::size_t spos = Path.find("/", bpos + 1);
      std::size_t length = Path.length();

      if (spos != std::string::npos) // if we found a "/"
        length = spos - bpos;

      std::string envVar = Path.substr(bpos + 1, length -1); //"HOME"
      const char* c_Path = getenv(envVar.c_str());
      std::string fullPath;
      if (c_Path != NULL) {
        fullPath = std::string(c_Path);
      } else {
        fullPath = std::string("");
      }
      Path.replace(bpos, length, fullPath);
      bpos = Path.find("$", bpos + 1); //search for next env variable
    }
  }

  typedef std::pair<bool, std::string> ParseResult_t;

  static ParseResult_t HandlePragmaHelper(Preprocessor &PP,
                                          const std::string &pragmaInst) {
    struct SkipToEOD_t {
      Preprocessor& m_PP;
      SkipToEOD_t(Preprocessor& PP): m_PP(PP) {}
      ~SkipToEOD_t() { m_PP.DiscardUntilEndOfDirective(); }
    } SkipToEOD(PP);

    Token Tok;
    PP.Lex(Tok);
    if (Tok.isNot(tok::l_paren)) {
      llvm::errs() << "cling:HandlePragmaHelper : expect '(' after #"
                   << pragmaInst;
      return ParseResult_t{false, ""};
    }
    std::string Literal;
    if (!PP.LexStringLiteral(Tok, Literal, pragmaInst.c_str(),
                             false /*allowMacroExpansion*/)) {
      // already diagnosed.
      return ParseResult_t {false, ""};
    }
    replaceEnvVars(Literal);

    return ParseResult_t {true, Literal};
  }

  class PHLoad: public PragmaHandler {
    Interpreter& m_Interp;

  public:
    PHLoad(Interpreter& interp):
      PragmaHandler("load"), m_Interp(interp) {}

    void HandlePragma(Preprocessor &PP,
                      PragmaIntroducerKind Introducer,
                      Token &FirstToken) override {
      // TODO: use Diagnostics!
      ParseResult_t Result = HandlePragmaHelper(PP, "pragma cling load");

      if (!Result.first)
        return;
      if (Result.second.empty()) {
        llvm::errs() << "Cannot load unnamed files.\n" ;
        return;
      }
      clang::Parser& P = m_Interp.getParser();
      Parser::ParserCurTokRestoreRAII savedCurToken(P);
      // After we have saved the token reset the current one to something which
      // is safe (semi colon usually means empty decl)
      Token& CurTok = const_cast<Token&>(P.getCurToken());
      CurTok.setKind(tok::semi);

      Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
      // We can't PushDeclContext, because we go up and the routine that pops
      // the DeclContext assumes that we drill down always.
      // We have to be on the global context. At that point we are in a
      // wrapper function so the parent context must be the global.
      TranslationUnitDecl* TU =
                  m_Interp.getCI()->getASTContext().getTranslationUnitDecl();
      Sema::ContextAndScopeRAII pushedDCAndS(m_Interp.getSema(),
                                             TU, m_Interp.getSema().TUScope);
      Interpreter::PushTransactionRAII pushedT(&m_Interp);

      m_Interp.loadFile(Result.second, true /*allowSharedLib*/);
    }
  };

  class PHAddIncPath: public PragmaHandler {
    Interpreter& m_Interp;

  public:
    PHAddIncPath(Interpreter& interp):
      PragmaHandler("add_include_path"), m_Interp(interp) {}

    void HandlePragma(Preprocessor &PP,
                      PragmaIntroducerKind Introducer,
                      Token &FirstToken) override {
      // TODO: use Diagnostics!
      ParseResult_t Result = HandlePragmaHelper(PP,
                                           "pragma cling add_include_path");
      //if the function HandlePragmaHelper returned false,
      if (!Result.first)
        return;
      if (!Result.second.empty())
        m_Interp.AddIncludePath(Result.second);
    }
  };

  class PHAddLibraryPath: public PragmaHandler {
    Interpreter& m_Interp;

  public:
    PHAddLibraryPath(Interpreter& interp):
      PragmaHandler("add_library_path"), m_Interp(interp) {}

    void HandlePragma(Preprocessor &PP,
                      PragmaIntroducerKind Introducer,
                      Token &FirstToken) override {
      // TODO: use Diagnostics!
      ParseResult_t Result = HandlePragmaHelper(PP,
                                         "pragma cling add_library_path");
      //if the function HandlePragmaHelper returned false,
      if (!Result.first)
        return;
      if (!Result.second.empty()) {
      // if HandlePragmaHelper returned success, this means that
      //it also returned the path to be included
        InvocationOptions& Opts = m_Interp.getOptions();
        Opts.LibSearchPath.push_back(Result.second);
      }
    }
  };
}

void cling::addClingPragmas(Interpreter& interp) {
  Preprocessor& PP = interp.getCI()->getPreprocessor();
  // PragmaNamespace / PP takes ownership of sub-handlers.
  PP.AddPragmaHandler("cling", new PHLoad(interp));
  PP.AddPragmaHandler("cling", new PHAddIncPath(interp));
  PP.AddPragmaHandler("cling", new PHAddLibraryPath(interp));
}
