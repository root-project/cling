//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: DeclCollector.h 46525 2012-10-13 15:04:49Z vvassilev $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_PUNCTUATION_LEXER_H
#define CLING_PUNCTUATION_LEXER_H

namespace llvm {
  class StringRef;
}

namespace cling {

  namespace tok {
    enum TokenKind {
      l_square,   // "["
      r_square,   // "]"
      l_paren,    // "("
      r_paren,    // ")"
      l_brace,    // "{"
      r_brace,    // "}"
      quote,      // """
      apostrophe, // "'"
      comma,      // ","
      eof,
      unknown
    };
  }

  class Token {
  private:
    tok::TokenKind kind;
    const char* bufStart;
    const char* bufEnd;
    void startToken() {
      bufStart = 0;
      kind = tok::unknown;
    }
  public:
    tok::TokenKind getKind() const { return kind; }
    unsigned getLength() const { return bufEnd - bufStart; }
    const char* getBufStart() const { return bufStart; }
    bool isNot(tok::TokenKind K) const { return kind != (unsigned) K; }


    friend class PunctuationLexer;
  };

  class PunctuationLexer {
  protected:
    const char* bufferStart;
    const char* bufferEnd;
    const char* curPos;
  public:
    PunctuationLexer(const char* bufStart) 
      : bufferStart(bufStart), curPos(bufStart)
    { }
    PunctuationLexer(llvm::StringRef input);

    bool LexPunctuator(Token& Result);
    bool LexEndOfFile(Token& Result);
  };

} //end namespace cling

#endif
