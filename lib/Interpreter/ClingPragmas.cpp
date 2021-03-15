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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/LiteralSupport.h"
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

    enum {
      kLoadFile,
      kAddLibrary,
      kAddInclude,
      // Put all commands that expand environment variables above this
      kExpandEnvCommands,

      // Put all commands that only take string literals above this
      kArgumentsAreLiterals,

      kOptimize,
      kInvalidCommand,
    };

    bool GetNextLiteral(Preprocessor& PP, Token& Tok, std::string& Literal,
                        unsigned Cmd, const char* firstTime = nullptr) const {
      Literal.clear();

      PP.Lex(Tok);
      if (Tok.isLiteral()) {
        if (clang::tok::isStringLiteral(Tok.getKind())) {
          SmallVector<Token, 1> StrToks(1, Tok);
          StringLiteralParser LitParse(StrToks, PP);
          if (!LitParse.hadError)
            Literal = LitParse.GetString();
        } else {
          llvm::SmallString<64> Buffer;
          Literal = PP.getSpelling(Tok, Buffer).str();
        }
      }
      else if (Tok.is(tok::comma))
        return GetNextLiteral(PP, Tok, Literal, Cmd);
      else if (firstTime) {
        if (Tok.is(tok::l_paren)) {
          if (Cmd < kArgumentsAreLiterals) {
            if (!PP.LexStringLiteral(Tok, Literal, firstTime,
                                     false /*allowMacroExpansion*/)) {
              // already diagnosed.
              return false;
            }
          } else {
            PP.Lex(Tok);
            llvm::SmallString<64> Buffer;
            Literal = PP.getSpelling(Tok, Buffer).str();
          }
        }
      }

      if (Literal.empty())
        return false;

      if (Cmd < kExpandEnvCommands)
        utils::ExpandEnvVars(Literal);

      return true;
    }

    void ReportCommandErr(Preprocessor& PP, const Token& Tok) {
      PP.Diag(Tok.getLocation(), diag::err_expected)
        << "load, add_library_path, or add_include_path";
    }

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

    void LoadCommand(Preprocessor& PP, Token& Tok, std::string Literal) {
      // No need to load libraries when not executing anything.
      if (m_Interp.isInSyntaxOnlyMode())
        return;

      // Need to parse them all until the end to handle the possible
      // #include statements that will be generated
      struct LibraryFileInfo {
        std::string FileName;
        SourceLocation StartLoc;
      };
      std::vector<LibraryFileInfo> FileInfos;
      FileInfos.push_back({std::move(Literal), Tok.getLocation()});
      while (GetNextLiteral(PP, Tok, Literal, kLoadFile))
        FileInfos.push_back({std::move(Literal), Tok.getLocation()});

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

      for (const LibraryFileInfo& FI : FileInfos) {
        // FIXME: Consider the case where the library static init section has
        // a call to interpreter parsing header file. It will suffer the same
        // issue as if we included the file within the pragma.
        if (m_Interp.loadLibrary(FI.FileName, true) != Interpreter::kSuccess) {
          const clang::DirectoryLookup *CurDir = nullptr;
          if (PP.getHeaderSearchInfo().LookupFile(FI.FileName, FI.StartLoc,
              /*isAngled*/ false, /*fromDir*/ nullptr, /*CurDir*/ CurDir, /*Includers*/ {},
              /*SearchPath*/ nullptr, /*RelativePath*/ nullptr, /*RequestingModule*/ nullptr,
              /*suggestedModule*/ nullptr, /*IsMapped*/ nullptr,
              /*IsFrameworkFound*/ nullptr, /*SkipCache*/ true, /*BuildSystemModule*/ false,
              /*OpenFile*/ false, /*CacheFailures*/ false)) {
            PP.Diag(FI.StartLoc, diag::err_expected)
              << FI.FileName + " to be a library, but it is not. If this is a source file, use `#include \"" + FI.FileName + "\"`";
          } else {
            PP.Diag(FI.StartLoc, diag::err_pp_file_not_found)
              << FI.FileName;
          }
          return;
        }
      }
    }

    void OptimizeCommand(const char* Str) {
      char* ConvEnd = nullptr;
      int OptLevel = std::strtol(Str, &ConvEnd, 10 /*base*/);
      if (!ConvEnd || ConvEnd == Str) {
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
      } else
        CO.OptLevel = OptLevel;
  }

  public:
    ClingPragmaHandler(Interpreter& interp):
      PragmaHandler("cling"), m_Interp(interp) {}

    void HandlePragma(Preprocessor& PP,
                      PragmaIntroducer /*Introducer*/,
                      Token& /*FirstToken*/) override {

      Token Tok;
      PP.Lex(Tok);
      SkipToEOD OnExit(PP, Tok);

      // #pragma cling(load, "A")
      if (Tok.is(tok::l_paren))
        PP.Lex(Tok);

      if (Tok.isNot(tok::identifier)) {
        ReportCommandErr(PP, Tok);
        return;
      }

      const StringRef CommandStr = Tok.getIdentifierInfo()->getName();
      const unsigned Command = GetCommand(CommandStr);
      assert(Command != kArgumentsAreLiterals && Command != kExpandEnvCommands);
      if (Command == kInvalidCommand) {
        ReportCommandErr(PP, Tok);
        return;
      }

      std::string Literal;
      if (!GetNextLiteral(PP, Tok, Literal, Command, CommandStr.data())) {
        PP.Diag(Tok.getLocation(), diag::err_expected_after)
          << CommandStr << "argument";
        return;
      }

      switch (Command) {
        case kLoadFile:
        return LoadCommand(PP, Tok, std::move(Literal));
        case kOptimize:
          return OptimizeCommand(Literal.c_str());

        default:
          do {
            if (Command == kAddLibrary)
              m_Interp.getOptions().LibSearchPath.push_back(std::move(Literal));
            else if (Command == kAddInclude)
              m_Interp.AddIncludePath(Literal);
          } while (GetNextLiteral(PP, Tok, Literal, Command));
          break;
      }
    }
  };
}

void cling::addClingPragmas(Interpreter& interp) {
  Preprocessor& PP = interp.getCI()->getPreprocessor();
  // PragmaNamespace / PP takes ownership of sub-handlers.
  PP.AddPragmaHandler(StringRef(), new ClingPragmaHandler(interp));
}
