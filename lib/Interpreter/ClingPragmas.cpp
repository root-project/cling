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
#include "clang/Parse/ParseDiagnostic.h"

#include <cstdlib>

using namespace cling;
using namespace clang;

namespace {
  class ClingPragmaHandler: public PragmaHandler {
    Interpreter& m_Interp;

    struct SkipToEOD {
      Preprocessor& m_PP;
      Token& m_Tok;
      SkipToEOD(Preprocessor& PParg, Token& Tok):
        m_PP(PParg), m_Tok(Tok) {
      }
      ~SkipToEOD() {
        // Can't use Preprocessor::DiscardUntilEndOfDirective, as we may
        // already be on an eod token
        while (!m_Tok.isOneOf(tok::eod, tok::eof))
          m_PP.LexUnexpandedToken(m_Tok);
      }
    };

    void ReportCommandErr(Preprocessor& PP, const Token& Tok) {
      PP.Diag(Tok.getLocation(), diag::err_expected)
        << "load, add_library_path, or add_include_path";
    }

    enum {
      kLoadFile,
      kAddLibrary,
      kAddInclude,
      // Put all commands that expand environment variables above this
      kExpandEnvCommands,
      kOptimize = kExpandEnvCommands,
      kInvalidCommand,
    };

    int GetCommand(const StringRef CommandStr) {
      if (CommandStr == "load")
        return kLoadFile;
      else if (CommandStr == "add_library_path")
        return kAddLibrary;
      else if (CommandStr == "add_include_path")
        return kAddInclude;
      else if (CommandStr == "optimize")
        return kOptimize;
      return kInvalidCommand;
    }
    
  public:
    ClingPragmaHandler(Interpreter& interp):
      PragmaHandler("cling"), m_Interp(interp) {}

    void HandlePragma(Preprocessor& PP,
                      PragmaIntroducerKind Introducer,
                      Token& FirstToken) override {

      Token Tok;
      PP.Lex(Tok);
      SkipToEOD OnExit(PP, Tok);

      if (Tok.isNot(tok::identifier)) {
        ReportCommandErr(PP, Tok);
        return;
      }

      const StringRef CommandStr = Tok.getIdentifierInfo()->getName();
      const int Command = GetCommand(CommandStr);
      if (Command == kInvalidCommand) {
        ReportCommandErr(PP, Tok);
        return;
      }

      PP.Lex(Tok);
      if (Tok.isNot(tok::l_paren)) {
        PP.Diag(Tok.getLocation(), diag::err_expected_lparen_after)
                << CommandStr;
        return;
      }

      std::string Literal;
      if (Command < kExpandEnvCommands) {
        if (!PP.LexStringLiteral(Tok, Literal, CommandStr.str().c_str(),
                                 false /*allowMacroExpansion*/)) {
          // already diagnosed.
          return;
        }
        utils::ExpandEnvVars(Literal);
      } else {
        PP.Lex(Tok);
        llvm::SmallString<64> Buffer;
        Literal = PP.getSpelling(Tok, Buffer).str();
      }

      switch (Command) {
        case kLoadFile:
          if (!m_Interp.isInSyntaxOnlyMode()) {
            // No need to load libraries if we're not executing anything.

            clang::Parser& P = m_Interp.getParser();
            Parser::ParserCurTokRestoreRAII savedCurToken(P);
            // After we have saved the token reset the current one to something
            // which is safe (semi colon usually means empty decl)
            Token& CurTok = const_cast<Token&>(P.getCurToken());
            CurTok.setKind(tok::semi);
            
            Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
            // We can't PushDeclContext, because we go up and the routine that
            // pops the DeclContext assumes that we drill down always.
            // We have to be on the global context. At that point we are in a
            // wrapper function so the parent context must be the global.
            TranslationUnitDecl* TU =
            m_Interp.getCI()->getASTContext().getTranslationUnitDecl();
            Sema::ContextAndScopeRAII pushedDCAndS(m_Interp.getSema(),
                                                  TU, m_Interp.getSema().TUScope);
            Interpreter::PushTransactionRAII pushedT(&m_Interp);
            
            m_Interp.loadFile(Literal, true /*allowSharedLib*/);
          }
          break;
        case kAddLibrary:
          m_Interp.getOptions().LibSearchPath.push_back(std::move(Literal));
          break;
        case kAddInclude:
          m_Interp.AddIncludePath(Literal);
          break;

        case kOptimize: {
          char* ConvEnd = nullptr;
          int OptLevel = std::strtol(Literal.c_str(), &ConvEnd, 10 /*base*/);
          if (!ConvEnd || ConvEnd == Literal.c_str()) {
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
            // Another #pragma already changed the opt level, a conflict that
            // cannot be resolve here.  Mention and keep the lower one.
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
      }
    }
  };
}

void cling::addClingPragmas(Interpreter& interp) {
  Preprocessor& PP = interp.getCI()->getPreprocessor();
  // PragmaNamespace / PP takes ownership of sub-handlers.
  PP.AddPragmaHandler(StringRef(), new ClingPragmaHandler(interp));
}
