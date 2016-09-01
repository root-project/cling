//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Lukasz Janyst <ljanyst@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/SourceNormalization.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include <utility>

using namespace clang;

namespace {
///\brief A Lexer that exposes preprocessor directives.
class MinimalPPLexer: public Lexer {
public:
  ///\brief Construct a Lexer from LangOpts and source.
  MinimalPPLexer(const LangOptions &LangOpts, llvm::StringRef source):
    Lexer(SourceLocation(), LangOpts,
          source.begin(), source.begin(), source.end()) {}

  bool inPPDirective() const { return ParsingPreprocessorDirective; }

  ///\brief Lex, forwarding to Lexer::LexFromRawLexer, and keeping track of
  /// preprocessor directives to provide a tok::eod corresponding to a
  /// tok::hash.
  bool Lex(Token& Tok) {
    bool ret = LexFromRawLexer(Tok);
    if (inPPDirective()) {
      // Saw a PP directive; probe for eod to end PP parsing mode.
      if (Tok.is(tok::eod))
        ParsingPreprocessorDirective = false;
    } else {
      if (Tok.is(tok::hash)) {
        // Found a PP directive, request tok::eod to be generated.
        ParsingPreprocessorDirective = true;
      }
    }
    return ret;
  }

  ///\brief Advance to token with given token kind.
  ///
  /// \param Tok - Token to advance.
  /// \param kind - Token kind where to stop lexing.
  /// \return - Result of most recent call to Lex().
  bool AdvanceTo(Token& Tok, tok::TokenKind kind) {
    while (!Lex(Tok)) {
      if (Tok.is(kind))
        return false;
    }
    return true;
  }

  ///\brief Lex a token that requires no cleaning
  ///
  /// \return - Result of the lex and whether the token is clean.
  bool LexClean(Token& Tok) {
    if (!LexFromRawLexer(Tok))
      return !Tok.needsCleaning();
    return false;
  }

  ///\brief Test if a given token is a valid identifier for the current language
  ///
  /// \param Tok - Token, advanced to first token to test
  /// \return - The valid identifier or empty llvm::StringRef
  llvm::StringRef Identifier(Token& Tok) const {
    if (Tok.is(tok::raw_identifier)) {
      StringRef Id(Tok.getRawIdentifier());
      if (Lexer::isIdentifierBodyChar(Id.front(), getLangOpts()))
        return Id;
    }
    return llvm::StringRef();
  }

  ///\brief Test if the given input is a function definition
  ///
  /// \param Tok - Token, advanced to first token to test
  /// \param First - First token identifier.
  /// \return - true if the input is a function
  bool IsFunction(Token& Tok, llvm::StringRef First) {
    if (!Lexer::isIdentifierBodyChar(First.front(), getLangOpts()))
      return false;

    // Early out calling a function/macro Ident()
    if (!LexClean(Tok) || Tok.is(tok::l_paren))
      return false;

    bool Ctor = false;
    if (getLangOpts().CPlusPlus && Tok.is(tok::coloncolon)) {
      // CLASS::CLASS or CLASS::~CLASS
      if (!LexClean(Tok))
        return false;
      if (Tok.is(tok::tilde)) {
        if (!LexClean(Tok))
          return false;
      } else
        Ctor = true;

      // Constructor and Desctructor identifiers must match
      if (!First.equals(Identifier(Tok)))
        return false;

      // Advance to argument list
      if (!LexClean(Tok))
        return false;
    } else {
      // This doesn't handle macro expansion. Missing anything?
      if (First.equals("static") || First.equals("constexpr") ||
          First.equals("inline") || First.equals("const")) {
        if (!LexClean(Tok))
          return false;
      }

      if (!Tok.is(tok::raw_identifier)) {
        // Skip over all *& tokens for a return value
        do {
          if (!Tok.isOneOf(tok::star, tok::amp))
            return false;
          if (!LexClean(Tok))
            return false;
        } while (!Tok.is(tok::raw_identifier));
      }

      // Function or class name
      if (Identifier(Tok).empty())
        return false;

      // Advance to argument list or method name
      if (!LexClean(Tok))
        return false;

      if (getLangOpts().CPlusPlus && Tok.is(tok::coloncolon)) {
        // Method name
        if (!LexClean(Tok) || Identifier(Tok).empty())
          return false;
        // Advance to argument list
        if (!LexClean(Tok))
          return false;
      }
    }

    // Argument list
    if (!Tok.is(tok::l_paren))
      return false;

    for (int unBalanced = 1; unBalanced;) {
      if (!LexClean(Tok))
        return false;
      if (Tok.is(tok::r_paren))
        --unBalanced;
      else if (Tok.is(tok::l_paren))
        ++unBalanced;
    }

    if (!LexClean(Tok))
      return false;

    // 'int func() {' or 'CLASS::method() {'
    if (Tok.is(tok::l_brace))
      return true;

    if (getLangOpts().CPlusPlus) {
      // constructor initialization 'CLASS::CLASS() :'
      if (Ctor && Tok.is(tok::colon))
        return true;

      // class const method 'CLASS::method() const {'
      if (!Ctor && Identifier(Tok).equals("const"))
        return LexClean(Tok) && Tok.is(tok::l_brace);
    }

    return false;
  }
};

size_t getFileOffset(const Token& Tok) {
  return Tok.getLocation().getRawEncoding();
}

}

size_t
cling::utils::isUnnamedMacro(llvm::StringRef source,
                             clang::LangOptions& LangOpts) {
  // Find the first token that is not a non-cpp directive nor a comment.
  // If that token is a '{' we have an unnamed macro.

  MinimalPPLexer Lex(LangOpts, source);
  Token Tok;
  bool AfterHash = false;
  while (true) {
    bool atEOF = Lex.Lex(Tok);

    if (atEOF)
      return std::string::npos;

    if (Lex.inPPDirective() || Tok.is(tok::eod)) {
      if (AfterHash) {
        if (Tok.is(tok::raw_identifier)) {
          StringRef keyword(Tok.getRawIdentifier());
          if (keyword.startswith("if")) {
            // This could well be
            //   #if FOO
            //   {
            // where we would determine this to be an unnamed macro and replace
            // '{' by ' ', whether FOO is #defined or not. Instead, assume that
            // this is not an unnamed macro and we need to parse it as is.
            return std::string::npos;
          }
        }
        AfterHash = false;
      } else
        AfterHash = Tok.is(tok::hash);

      continue; // Skip PP directives.
    }

    if (Tok.is(tok::l_brace))
      return getFileOffset(Tok);

    if (Tok.is(tok::comment))
      continue; // ignore comments

    return std::string::npos;
  }

  // Empty file?

  return std::string::npos;
}



size_t cling::utils::getWrapPoint(std::string& source,
                                  const clang::LangOptions& LangOpts) {
  // TODO: For future reference.
  // Parser* P = const_cast<clang::Parser*>(m_IncrParser->getParser());
  // Parser::TentativeParsingAction TA(P);
  // TPResult result = P->isCXXDeclarationSpecifier();
  // TA.Revert();
  // return result == TPResult::True();

  MinimalPPLexer Lex(LangOpts, source);
  Token Tok;

  //size_t wrapPoint = 0;

  while (true) {
    bool atEOF = Lex.Lex(Tok);
    if (Lex.inPPDirective() || Tok.is(tok::eod)) {
      //wrapPoint = getFileOffset(Tok);
      if (atEOF)
        break;
      continue; // Skip PP directives; they just move the wrap point.
    }

    if (Tok.is(tok::eof)) {
      // Reached EOF before seeing a non-preproc token.
      // Nothing to wrap.
      return std::string::npos;
    }

    const tok::TokenKind kind = Tok.getKind();

    if (kind == tok::raw_identifier && !Tok.needsCleaning()) {
      StringRef keyword(Tok.getRawIdentifier());
      if (keyword.equals("using")) {
        // FIXME: Using definitions and declarations should be decl extracted.
        // Until we have that, don't wrap them if they are the only input.
        if (Lex.AdvanceTo(Tok, tok::semi)) {
          // EOF while looking for semi. Don't wrap.
          return std::string::npos;
        }
        // There is "more" - let's assume this input consists of a using
        // declaration or definition plus some code that should be wrapped.
        return getFileOffset(Tok);
      }
      if (keyword.equals("extern"))
        return std::string::npos;
      if (keyword.equals("namespace"))
        return std::string::npos;
      if (keyword.equals("template"))
        return std::string::npos;
      if (Lex.IsFunction(Tok, keyword))
        return std::string::npos;

      // There is something else here that needs to be wrapped.
      return getFileOffset(Tok);
    }

    // FIXME: in the future, continue lexing to extract relevant PP directives;
    // return wrapPoint
    // There is something else here that needs to be wrapped.
    return getFileOffset(Tok);
  }

  // We have only had PP directives; no need to wrap.
  return std::string::npos;
}
