//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_META_LEXER_H
#define CLING_META_LEXER_H

#include "llvm/ADT/StringRef.h"

namespace cling {

  namespace tok {
    enum TokenKind {
      l_square,   // "["
      r_square,   // "]"
      l_paren,    // "("
      r_paren,    // ")"
      l_brace,    // "{"
      r_brace,    // "}"
      stringlit,  // ""...""
      charlit,    // "'.'"
      comma,      // ","
      dot,        // "."
      excl_mark,  // "!"
      quest_mark, // "?"
      slash,      // "/"
      backslash,  // "\"
      greater,    // ">"
      ampersand,  // "&"
      hash,       // "#"
      ident,      // (a-zA-Z)[(0-9a-zA-Z)*]
      raw_ident,  // .*^(' '|'\t')
      comment,    // //
      space,      // (' ' | '\t')*
      constant,   // {0-9}
      at,         // @
      asterik,    // *
      semicolon,  // ;
      eof,
      unknown
    };
  }

  class Token {
  private:
    tok::TokenKind kind;
    const char* bufStart;
    unsigned length;
    mutable unsigned value;
  public:
    Token(const char* Buf = nullptr) { startToken(Buf); }

    void startToken(const char* Pos = nullptr) {
      kind = tok::unknown;
      bufStart = Pos;
      value = ~0U;
      length = 0;
    }
    tok::TokenKind getKind() const { return kind; }
    void setKind(tok::TokenKind K) { kind = K; }
    unsigned getLength() const { return length; }
    void setLength(unsigned L) { length = L; }
    const char* getBufStart() const { return bufStart; }
    void setBufStart(const char* Pos) { bufStart = Pos; }

    bool isNot(tok::TokenKind K) const { return kind != K; }
    bool is(tok::TokenKind K) const { return kind == K; }

    llvm::StringRef getIdent() const;
    llvm::StringRef getIdentNoQuotes() const {
      if (getKind() >= tok::stringlit && getKind() <= tok::charlit)
        return getIdent().drop_back().drop_front();
      return getIdent();
    }
    bool getConstantAsBool() const;
    unsigned getConstant() const;
  };

  class MetaLexer {
  protected:
    const char* bufferStart;
    const char* curPos;
  public:
    MetaLexer(llvm::StringRef input, bool skipWhiteSpace = false);
    void reset(llvm::StringRef Line);

    void Lex(Token& Tok);
    void LexAnyString(Token& Tok);

    static void LexPunctuator(const char* C, Token& Tok);
    // TODO: Revise. We might not need that.
    static bool LexPunctuatorAndAdvance(const char*& curPos, Token& Tok);
    static void LexQuotedStringAndAdvance(const char*& curPos, Token& Tok);
    void LexConstant(char C, Token& Tok);
    void LexIdentifier(char C, Token& Tok);
    void LexEndOfFile(char C, Token& Tok);
    void LexWhitespace(char C, Token& Tok);
    void SkipWhitespace();
    const char* getLocation() const { return curPos; }
  };
} //end namespace cling

#endif // CLING_PUNCTUATION_LEXER_H
