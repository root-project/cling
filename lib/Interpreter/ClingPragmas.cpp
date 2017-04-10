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
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/Output.h"
#include "cling/Utils/Paths.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Parse/Parser.h"

#include <cstdlib>

using namespace cling;
using namespace clang;

namespace {
  typedef std::pair<bool, std::string> ParseResult_t;

  static ParseResult_t HandlePragmaHelper(Preprocessor &PP,
                                          const std::string &pragmaInst,
                                          bool stringLiteralArg = true) {
    struct SkipToEOD_t {
      Preprocessor& m_PP;
      SkipToEOD_t(Preprocessor& PParg): m_PP(PParg) {}
      ~SkipToEOD_t() { m_PP.DiscardUntilEndOfDirective(); }
    } SkipToEOD(PP);

    Token Tok;
    PP.Lex(Tok);
    if (Tok.isNot(tok::l_paren)) {
      cling::errs() << "cling::HandlePragmaHelper: expect '(' after #"
                    << pragmaInst << '\n';
      return ParseResult_t{false, ""};
    }
    std::string Literal;
    if (stringLiteralArg) {
      if (!PP.LexStringLiteral(Tok, Literal, pragmaInst.c_str(),
                               false /*allowMacroExpansion*/)) {
        // already diagnosed.
        return ParseResult_t {false, ""};
      }
      utils::ExpandEnvVars(Literal);
    } else {
      // integer literal
      PP.Lex(Tok);
      if (!Tok.isLiteral()) {
        cling::errs()
          << "cling::HandlePragmaHelper: expect integer literal after #"
          << pragmaInst << '\n';
        return ParseResult_t {false, ""};
      }
      Literal = std::string(Tok.getLiteralData(), Tok.getLength());
    }

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
        cling::errs() << "Cannot load unnamed files.\n" ;
        return;
      }
      clang::Parser& P = m_Interp.getParser();
      Parser::ParserCurTokRestoreRAII savedCurToken(P);
      // After we have saved the token reset the current one to something which
      // is safe (semi colon usually means empty decl)
      Token& CurTok = const_cast<Token&>(P.getCurToken());
      CurTok.setKind(tok::semi);

      if (!m_Interp.isInSyntaxOnlyMode()) {
        // No need to load libraries if we're not executing anything.

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

  class PHOptLevel: public PragmaHandler {
    Interpreter& m_Interp;

  public:
    PHOptLevel(Interpreter& interp):
      PragmaHandler("optimize"), m_Interp(interp) {}

    void HandlePragma(Preprocessor &PP,
                      PragmaIntroducerKind Introducer,
                      Token &FirstToken) override {
      // TODO: use Diagnostics!
      ParseResult_t Result = HandlePragmaHelper(PP, "pragma cling optimize",
                                                false /*string literal*/);
      //if the function HandlePragmaHelper returned false,
      if (!Result.first)
        return;
      if (Result.second.empty()) {
        cling::errs() << "Missing optimization level.\n" ;
        return;
      }

      char* ConvEnd = nullptr;
      int OptLevel = std::strtol(Result.second.c_str(), &ConvEnd, 10 /*base*/);
      if (!ConvEnd || ConvEnd == Result.second.c_str()) {
        cling::errs() << "cling::PHOptLevel: "
          "missing or non-numerical optimization level.\n" ;
        return;
      }
      auto T = const_cast<Transaction*>(m_Interp.getCurrentTransaction());
      assert(T && "Parsing code without transaction!");
      // The topmost Transaction drives the jitting.
      T = T->getTopmostParent();
      CompilationOptions& CO = T->getCompilationOpts();
      if (CO.OptLevel != m_Interp.getDefaultOptLevel()) {
        // Another #pragma already changed the opt level. That's a conflict that
        // we cannot resolve here; mention that and  keep the lower one.
        cling::errs() << "cling::PHOptLevel: "
          "conflicting `#pragma cling optimize` directives: "
          "was already set to " << CO.OptLevel << '\n';
        if (CO.OptLevel > OptLevel) {
          CO.OptLevel = OptLevel;
          cling::errs() << "Setting to lower value of " << OptLevel << '\n';
        } else {
          cling::errs() << "Ignoring higher value of " << OptLevel << '\n';
        }
      } else {
        CO.OptLevel = OptLevel;
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
  PP.AddPragmaHandler("cling", new PHOptLevel(interp));
}
