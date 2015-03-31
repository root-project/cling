//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "MetaLexer.h"

#include "llvm/ADT/StringRef.h"

namespace cling {

  llvm::StringRef Token::getIdent() const {
    assert((is(tok::ident) || is(tok::raw_ident)) && "Token not an ident.");
    return llvm::StringRef(bufStart, getLength());
  }

  bool Token::getConstantAsBool() const {
    assert(kind == tok::constant && "Not a constant");
    return getConstant() != 0;
  }

  static unsigned int pow10[10] = { 1, 10, 100, 1000, 10000,
                                    100000, 1000000, 10000000, ~0U};
  unsigned Token::getConstant() const {
    assert(kind == tok::constant && "Not a constant");
    if (value == ~0U) {
      value = 0;
      //calculate the value
      for (size_t i = 0, e = length; i < e; ++i)
        value += (*(bufStart+i) -'0') * pow10[length - i - 1];
    }
    return value;
  }

  MetaLexer::MetaLexer(llvm::StringRef line)
    : bufferStart(line.data()), curPos(line.data())
  { }

  void MetaLexer::Lex(Token& Tok) {
    Tok.startToken(curPos);
    char C = *curPos++;
    switch (C) {
    case '"': case '\'':
      return LexQuotedStringAndAdvance(--curPos, Tok);
    case '[': case ']': case '(': case ')': case '{': case '}':
    case '\\': case ',': case '.': case '!': case '?': case '>':
    case '&': case '#': case '@':
      // INTENTIONAL FALL THROUGHs
      return LexPunctuator(C, Tok);

    case '/':
      if (*curPos != '/')
        return LexPunctuator(C, Tok);
      else {
        ++curPos;
        Tok.setKind(tok::comment);
        Tok.setLength(2);
        return;
      }

    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
      return LexConstant(C, Tok);

    case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
    case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
    case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
    case 'V': case 'W': case 'X': case 'Y': case 'Z':
    case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
    case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
    case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
    case 'v': case 'w': case 'x': case 'y': case 'z':
    case '_':
      return LexIdentifier(C, Tok);
    case ' ': case '\t':
      return LexWhitespace(C, Tok);
    case '\0':
      return LexEndOfFile(C, Tok);
    }
  }

  void MetaLexer::LexAnyString(Token& Tok) {
    Tok.startToken(curPos);

    // consume until we reach one of the "AnyString" delimiters or EOF.
    while(*curPos != ' ' && *curPos != '\t' && *curPos != '\0') {
      curPos++;
    }

    assert(Tok.getBufStart() != curPos && "It must consume at least on char");

    Tok.setKind(tok::raw_ident);
    Tok.setLength(curPos - Tok.getBufStart());
  }

  void MetaLexer::LexPunctuator(char C, Token& Tok) {
    Tok.setLength(1);
    switch (C) {
    case '['  : Tok.setKind(tok::l_square); break;
    case ']'  : Tok.setKind(tok::r_square); break;
    case '('  : Tok.setKind(tok::l_paren); break;
    case ')'  : Tok.setKind(tok::r_paren); break;
    case '{'  : Tok.setKind(tok::l_brace); break;
    case '}'  : Tok.setKind(tok::r_brace); break;
    case '"'  : Tok.setKind(tok::quote); break;
    case '\'' : Tok.setKind(tok::apostrophe); break;
    case ','  : Tok.setKind(tok::comma); break;
    case '.'  : Tok.setKind(tok::dot); break;
    case '!'  : Tok.setKind(tok::excl_mark); break;
    case '?'  : Tok.setKind(tok::quest_mark); break;
    case '/'  : Tok.setKind(tok::slash); break;
    case '\\' : Tok.setKind(tok::backslash); break;
    case '>'  : Tok.setKind(tok::greater); break;
    case '@' : Tok.setKind(tok::at); break;
    case '&'  : Tok.setKind(tok::ampersand); break;
    case '#'  : Tok.setKind(tok::hash); break;
    case '\0' : Tok.setKind(tok::eof); Tok.setLength(0); break;// if static call
    default: Tok.setLength(0); break;
    }
  }

  void MetaLexer::LexPunctuatorAndAdvance(const char*& curPos, Token& Tok,
                                          bool skipComments /*false*/) {
    Tok.startToken(curPos);
    while (true) {
      // On comment skip until the eof token.
      if (!skipComments && curPos[0] == '/' && curPos[1] == '/') {
        while (*curPos != '\0' && *curPos != '\r' && *curPos != '\n')
          ++curPos;
        if (*curPos == '\0') {
          Tok.setBufStart(curPos);
          Tok.setKind(tok::eof);
          Tok.setLength(0);
          return;
        }
      }
      MetaLexer::LexPunctuator(*curPos++, Tok);
      if (Tok.isNot(tok::unknown))
        return;
    }
  }

 void MetaLexer::LexQuotedStringAndAdvance(const char*& curPos, Token& Tok) {
    // Tok must be the starting quote (single or double), and we will
    // lex until the next one or the end of the line.

    const char* const start = curPos;
    LexPunctuator(*curPos++, Tok); // '"' or "'"
    assert( (Tok.getKind() >= tok::quote && Tok.getKind() <= tok::apostrophe) );
    tok::TokenKind EndTokKind = Tok.getKind();
    Tok.setKind(tok::raw_ident);

    //consuming the string
    while (true) {

      // "AB\"foo \0x12 \012 \n"
      /*  bool escape = false;
      while ( (escape || *curPos != *start) && *curPos != '\0' && *curPos != '\r' && *curPos != '\n') {
        escape = ( (*curPos) == '\\' );
        ++curPos;
      }*/
      if (*curPos == '\0') {
        Tok.setBufStart(curPos);
        Tok.setKind(tok::eof);
        Tok.setLength(0);
        return;
      }
      MetaLexer::LexPunctuator(*curPos++, Tok);
      if (Tok.is(EndTokKind)) {
        Tok.setLength(curPos - start + 1);
        return;
      }
    }
  }

  void MetaLexer::LexConstant(char C, Token& Tok) {
    while (C >= '0' && C <= '9')
      C = *curPos++;

    --curPos; // Back up over the non ident char.
    Tok.setLength(curPos - Tok.getBufStart());
    Tok.setKind(tok::constant);
  }

  void MetaLexer::LexIdentifier(char C, Token& Tok) {
    while (C == '_' || (C >= 'A' && C <= 'Z') || (C >= 'a' && C <= 'z')
           || (C >= '0' && C <= '9'))
      C = *curPos++;

    --curPos; // Back up over the non ident char.
    Tok.setLength(curPos - Tok.getBufStart());
    if (Tok.getLength())
      Tok.setKind(tok::ident);
  }

  void MetaLexer::LexEndOfFile(char C, Token& Tok) {
    if (C == '\0') {
      Tok.setKind(tok::eof);
      Tok.setLength(1);
    }
  }

  void MetaLexer::LexWhitespace(char C, Token& Tok) {
    while((C == ' ' || C == '\t') && C != '\0')
      C = *curPos++;

    --curPos; // Back up over the non whitespace char.
    Tok.setLength(curPos - Tok.getBufStart());
    Tok.setKind(tok::space);
  }
} // end namespace cling
