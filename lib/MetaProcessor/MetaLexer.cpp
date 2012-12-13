//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "MetaLexer.h"

#include "llvm/ADT/StringRef.h"

namespace cling {

  llvm::StringRef Token::getIdent() const {
    assert((is(tok::ident) || is(tok::raw_ident)) && "Token not an ident.");
    return llvm::StringRef(bufStart, getLength());
  }

  bool Token::getConstant() const {
    assert(kind == tok::constant && "Not a constant");
    return *bufStart == '1';
  }

  MetaLexer::MetaLexer(llvm::StringRef line) 
    : bufferStart(line.data()), curPos(line.data()) 
  { }

  void MetaLexer::Lex(Token& Tok) {
    Tok.startToken(curPos);
    char C = *curPos++;
    switch (C) {
    case '[': case ']': case '(': case ')': case '{': case '}': case '"':
    case '\'': case '\\': case ',': case '.': case '!':
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

    case '0': case '1':/* case '2': case '3': case '4':
                          case '5': case '6': case '7': case '8': case '9':*/
      Tok.setKind(tok::constant);
      Tok.setLength(1);
      return;

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
    case '/'  : Tok.setKind(tok::slash); break;
    case '\\'  : Tok.setKind(tok::backslash); break;
    case '\0' : Tok.setKind(tok::eof); Tok.setLength(0); break;// if static call
    default: Tok.setLength(0); break;
    }
  }

  void MetaLexer::LexPunctuatorAndAdvance(const char*& curPos, Token& Tok) {
    Tok.startToken(curPos);
    while (true) {
      // On comment skip until the eof token.
      if (curPos[0] == '/' && curPos[1] == '/') {
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
