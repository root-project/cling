//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <Axel.Naumann@cern.ch>
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

  const LangOptions &LangOpts;

  ///\brief Jump to the next token after a scope chain A::B::C::D
  ///
  bool SkipScopes(Token& Tok) {
    Token LastTok;
    return SkipScopes(Tok, LastTok);
  }

  bool SkipScopes(Token& Tok, Token& LastTok) {
    LastTok = Tok;

    if (LangOpts.CPlusPlus) {
      while (Tok.is(tok::coloncolon)) {
        if (!LexClean(LastTok) || Identifier(LastTok).empty())
          return false;
        if (!LexClean(Tok))
          return false;
      }
    }
    return true;
  }

  ///\brief Skips all contiguous '*' '&' tokens
  ///
  bool SkipPointerRefs(Token& Tok) {
    while (Tok.isOneOf(tok::star, tok::amp)) {
      if (!LexClean(Tok))
        return false;
    }
    return true;
  }

  ///\brief Test if a given token is a valid identifier for the current language
  ///
  /// \param Tok - Token, advanced to first token to test
  /// \return - The valid identifier or empty llvm::StringRef
  llvm::StringRef Identifier(Token& Tok) const {
    if (Tok.is(tok::raw_identifier)) {
      StringRef Id(Tok.getRawIdentifier());
      if (Lexer::isAsciiIdentifierContinueChar(Id.front(), LangOpts))
        return Id;
    }
    return llvm::StringRef();
  }

  ///\brief Skip an identifier of a function.
  bool SkipIdentifier(Token& Tok) {
    if (Tok.isNot(tok::raw_identifier)) {
      // If we're not at an identifier, we might be still be in return value:
      // A::B::C funcname() or int * funcname()
      if (!SkipScopes(Tok))
        return false;
      if (!SkipPointerRefs(Tok))
        return false;
    }

    auto LastTok{Tok};
    // skip scopes in function name
    // e.g. int ::the_namespace::class_a::mem_func() { return 3; }
    if ((Tok.is(tok::coloncolon) && !SkipScopes(Tok, LastTok)) ||
        (!LexClean(Tok) ||
         (Tok.is(tok::coloncolon) && !SkipScopes(Tok, LastTok))))
      return false;

    const auto Ident{Identifier(LastTok)}; // function, operator or class name

    if (Ident.empty())
      return false;

    if (Ident.equals("operator")) {
      // Tok is the operator, e.g. <=, ==, +...
      // however, for operator() and [], Tok only contains the
      // left side so we need to parse the closing right side
      // TODO: tok::spaceship operator<=>
      if ((Tok.isOneOf(tok::l_paren, tok::l_square) && !CheckBalance(Tok)) ||
          // user-defined literal, e.g. double operator "" _dd(long double t)
          // TODO: parse without the space right after "",
          // e.g. double operator ""_dd(long double t)
          (Tok.is(tok::string_literal) && !LexClean(Tok)) ||
          // Advance to argument list or method name
          !LexClean(Tok))
        return false;
    }

    if (!SkipScopes(Tok))
      return false;

    return true;
  }

public:
  ///\brief Construct a Lexer from LangOpts and source.
  MinimalPPLexer(const LangOptions &LOpts, llvm::StringRef source):
    Lexer(SourceLocation(), LOpts,
          source.begin(), source.begin(), source.end()),
    LangOpts(LOpts) {}

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

  ///\brief Make sure a token is closed/balanced properly
  ///
  bool CheckBalance(Token& Tok) {
    const tok::TokenKind In = Tok.getKind();
    const tok::TokenKind Out
      = In == tok::less ? tok::greater : tok::TokenKind(In + 1);
    assert((In == tok::l_paren || In == tok::l_brace || In == tok::l_square
            || In == tok::less) && "Invalid balance token");
    bool atEOF = false;
    int  unBalanced = 1;
    while (unBalanced && !atEOF) {
      atEOF = !LexClean(Tok);
      if (Tok.is(Out))
        --unBalanced;
      else if (Tok.is(In))
        ++unBalanced;
    }
    return unBalanced == 0;
  }

  enum DefinitionType {
    kNONE,     ///< Neither a function or class defintion
    kFunction, ///< Function, method, constructor, or destructor definition
    kClass     ///< Class definition
  };

  ///\brief Test if the given input is a function or class definition
  ///
  /// \param Tok - Token, advanced to first token to test
  /// \param First - First token identifier.
  /// \param[out] HasBody - if set to `true`, the function/class body follows;
  ///                       thus, the caller needs to consume tokens until the
  ///                       closing `}`
  /// \return - Typeof definition, function/method or class
  DefinitionType IsClassOrFunction(Token& Tok, llvm::StringRef First,
                                   bool& HasBody) {
    HasBody = true;
    /// ###TODO: Allow preprocessor expansion
    if (!Lexer::isAsciiIdentifierContinueChar(First.front(), LangOpts))
      return kNONE;

    // Early out calling a function/macro Ident()
    if (!LexClean(Tok) || Tok.is(tok::l_paren))
      return kNONE;

    bool Ctor = false;
    bool Dtor = false;

    if (LangOpts.CPlusPlus && Tok.is(tok::coloncolon)) {
      // CLASS::CLASS() or CLASS::~CLASS()
      // CLASS::NESTED::NESTED()
      // CLASS::type func()
      // CLASS::NESTED::type Var();
      llvm::StringRef Ident;

      do {
        if (!LexClean(Tok))
          return kNONE;
        if (Tok.is(tok::raw_identifier)) {
          if (!Ident.empty())
            First = Ident;
          Ident = Identifier(Tok);
          if (!LexClean(Tok))
            return kNONE;
          if (Tok.is(tok::less)) {
            // A template: Ident <
            if (!CheckBalance(Tok))
              return kNONE;
            if (!LexClean(Tok))
              return kNONE;
          }
        }
      } while (Tok.is(tok::coloncolon));

      if (Tok.is(tok::tilde)) {
        Dtor = true;

        if (!LexClean(Tok))
          return kNONE;
        if (!Ident.empty())
          First = Ident;
        Ident = Identifier(Tok);

        // Advance to argument list
        if (Ident.empty() || !LexClean(Tok))
          return kNONE;
      } else
        Ctor = true;

      // Constructor and Destructor identifiers must match
      if (!First.equals(Ident)) {
        if (!SkipPointerRefs(Tok))
          return kNONE;

        // Advance to argument list, or next scope
        if (!SkipIdentifier(Tok))
          return kNONE;

        // Function name should be last on scope chain
        if (!SkipScopes(Tok))
          return kNONE;

        Ctor = false;
        Dtor = false;
      }
    } else {
      bool SeenModifier = false;
      auto AnalyzeModifier = [&SeenModifier](llvm::StringRef Modifier) {
        if (Modifier.equals("signed") || Modifier.equals("unsigned") ||
            Modifier.equals("short") || Modifier.equals("long")) {
          SeenModifier = true;
        }
      };
      if (First.equals("struct") || First.equals("class")) {
        do {
          // Identifier(Tok).empty() is redundant 1st time, but simplifies code
          if (Identifier(Tok).empty() || !LexClean(Tok))
            return kNONE;
        } while (LangOpts.CPlusPlus && Tok.is(tok::coloncolon) &&
                 LexClean(Tok));

        // 'class T {' 'struct T {' 'class T :'
        if (Tok.is(tok::l_brace))
          return kClass;
        if (Tok.is(tok::colon))
          return !AdvanceTo(Tok, tok::l_brace) ? kClass : kNONE;

      } else if (First.equals("static") || First.equals("constexpr") ||
                 First.equals("inline") || First.equals("const")) {
        // First check if the current keyword is a modifier.
        llvm::StringRef Modifier = Identifier(Tok);
        AnalyzeModifier(Modifier);

        // Advance past keyword for below
        if (!LexClean(Tok))
          return kNONE;
      } else {
        AnalyzeModifier(First);
      }

      if (!SkipIdentifier(Tok))
        return kNONE;

      // If we have not yet reached the argument list and seen a modifier
      // keyword, try to skip this once.
      if (SeenModifier && Tok.isNot(tok::l_paren) && !SkipIdentifier(Tok))
        return kNONE;
    }

    // Argument list
    if (Tok.isNot(tok::l_paren))
      return kNONE;

    // ##TODO
    // Lex the argument identifiers so we can know if this is a declaration
    // RVAL IDENT(A,B,C)          -> could be:
    //
    // T inst(a,b,c);            -> class instance
    // T func(T0 a, T1 b, T2 c); -> func declaration
    //
    // Without macro expansion it's difficult to distinguish cases, but as the
    // detection can fail because of macros already, would it be enough to check
    // that there are two idents not separated by commas between parenthesis?
    // It still wouldn't work for RVAL IDENT(), but -Wno-vexing-parse could be
    // passed to clang in initialization.
    //
    // Maybe the best would be to lookup the Decl IDENT to see if its a class?

    if (!CheckBalance(Tok))
      return kNONE;

    if (!LexClean(Tok))
      return kNONE;

    // 'int func() {' or 'CLASS::method() {'
    if (Tok.is(tok::l_brace))
      return kFunction;

    if (LangOpts.CPlusPlus) {
      // constructor initialization 'CLASS::CLASS() :'
      if (Ctor && Tok.is(tok::colon))
        return !AdvanceTo(Tok, tok::l_brace) ? kFunction : kNONE;

      if (Ctor || Dtor) {
        // e.g. CLASS::CLASS() = default;
        //      CLASS::~CLASS();
        if (Tok.is(tok::equal)) {
          if ((!LexClean(Tok) && Tok.isNot(tok::raw_identifier)) ||
              (!LexClean(Tok) && Tok.isNot(tok::semi)))
            return kNONE;

          HasBody = false;
          return kFunction;
        }
      }

      // class const method 'CLASS::method() const {'
      if (!Ctor && Identifier(Tok).equals("const")) {
        if (LexClean(Tok) && Tok.is(tok::l_brace))
          return kFunction;
      }
    }

    return kNONE;
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
          if (keyword.starts_with("if")) {
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

  while (true) {
    bool atEOF = Lex.Lex(Tok);
    if (Lex.inPPDirective() || Tok.is(tok::eod)) {
      if (atEOF)
        break;
      continue; // Skip PP directives; they just move the wrap point.
    }

    if (Tok.is(tok::eof)) {
      // Reached EOF before seeing a non-preproc token.
      // Nothing to wrap.
      return std::string::npos;
    }

    // detect the attribute (__global__, __device__ and __host__) of CUDA
    // kernels at the beginning of a function definition
    // FIXME: should replaced by a generic solution
    if (LangOpts.CUDA) {
      do {
        if (Tok.getKind() == tok::raw_identifier) {
          StringRef keyword(Tok.getRawIdentifier());
          if (keyword.equals("__global__") || keyword.equals("__device__") ||
              keyword.equals("__host__"))
            // if attribute was found, skip the token and use the function
            // detection later
            Lex.Lex(Tok);
          else
            break;
        } else
          break;
      } while (true);
    }

    // Prior behavior was to return getFileOffset, which was only used as an
    // in a test against std::string::npos. By returning 0 we preserve prior
    // behavior to pass the test against std::string::npos and wrap everything
    const size_t offset = 0;

    // Check, if a function with c++ attributes should be defined.
    while (Tok.getKind() == tok::l_square) {
      Lex.Lex(Tok);
      // Check, if attribute starts with '[['
      if (Tok.getKind() != tok::l_square) {
        return offset;
      }
      // Check, if the second '[' is closing.
      if (!Lex.CheckBalance(Tok)) {
        return offset;
      }
      Lex.Lex(Tok);
      // Check, if the first '[' is closing.
      if (Tok.getKind() != tok::r_square) {
        return offset;
      }
      Lex.Lex(Tok);
    }

    if (Tok.getKind() == tok::coloncolon)
      Lex.LexClean(Tok);

    if (Tok.getKind() == tok::raw_identifier && !Tok.needsCleaning()) {
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
        //
        // We need to include the ';' in the offset as this will be a
        // non-wrapped statement.
        return getFileOffset(Tok) + 1;
      }
      if (keyword.equals("extern"))
        return std::string::npos;
      if (keyword.equals("namespace"))
        return std::string::npos;
      if (keyword.equals("template"))
        return std::string::npos;

      auto HasBody{false};

      if (const MinimalPPLexer::DefinitionType T =
              Lex.IsClassOrFunction(Tok, keyword, HasBody)) {
        if (HasBody) {
          assert(Tok.is(tok::l_brace) && "Lexer begin location invalid");

          if (!Lex.CheckBalance(Tok))
            return offset;

          assert(Tok.is(tok::r_brace) && "Lexer end location invalid");
        }

        const size_t rBrace = getFileOffset(Tok);
        // Wrap everything after '}'
        atEOF = !Lex.LexClean(Tok);
        bool hadSemi = Tok.is(tok::semi);
        size_t wrapPoint = getFileOffset(Tok);
        if (!atEOF) {
          if (hadSemi) {
            atEOF = !Lex.LexClean(Tok);
            if (!atEOF) {
              // Wrap everything after ';'
              wrapPoint = getFileOffset(Tok);
            }
          } else if (T == MinimalPPLexer::kClass) {
            // 'struct T {} t     '
            // 'struct E {} t = {}'
            // Value print: We want to preserve Tok.is(tok::raw_identifier)
            // unless the statement was terminated by a semi-colon anyway.
            Token Tok2;
            atEOF = Lex.AdvanceTo(Tok2, tok::semi);
            if ((hadSemi = Tok2.is(tok::semi)))
              Tok = Tok2;
          }
        }

        // If nothing left to lex, then don't wrap any of it
        if (atEOF) {
          if (T == MinimalPPLexer::kClass) {
            if (!hadSemi) {
              // Support lack of semi-colon value printing 'struct T {} t'
              if (Tok.is(tok::raw_identifier))
                return 0;
              if (!LangOpts.HeinousExtensions) {
                // Let's fix 'class NoTerminatingSemi { ... }' for them!
                // ### TODO DiagnosticOptions.ShowFixits might be better
                source.insert(rBrace+1, ";");
                return source.size();
              }
            }
          }
          return std::string::npos;
        }

        return wrapPoint;
      }

      // There is something else here that needs to be wrapped.
      return offset;
    }

    // FIXME: in the future, continue lexing to extract relevant PP directives;
    // return wrapPoint
    // There is something else here that needs to be wrapped.
    return offset;
  }

  // We have only had PP directives; no need to wrap.
  return std::string::npos;
}
