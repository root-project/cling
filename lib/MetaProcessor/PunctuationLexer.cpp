//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: DeclCollector.cpp 47416 2012-11-18 22:44:58Z vvassilev $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "PunctuationLexer.h"

#include "llvm/ADT/StringRef.h"

namespace cling {

  PunctuationLexer::PunctuationLexer(llvm::StringRef line) 
    : bufferStart(line.data()), curPos(line.data()) 
  { }

  bool PunctuationLexer::LexPunctuator(Token& Tok) {
    Tok.startToken();
    while (true) {
      Tok.bufStart = curPos;
      switch (*curPos++) {
      case '['  : Tok.kind = tok::l_square; Tok.bufEnd = curPos; return true;
      case ']'  : Tok.kind = tok::r_square; Tok.bufEnd = curPos; return true;
      case '('  : Tok.kind = tok::l_paren; Tok.bufEnd = curPos; return true;
      case ')'  : Tok.kind = tok::r_paren; Tok.bufEnd = curPos; return true;
      case '{'  : Tok.kind = tok::l_brace; Tok.bufEnd = curPos; return true;
      case '}'  : Tok.kind = tok::r_brace; Tok.bufEnd = curPos; return true;
      case '"'  : Tok.kind = tok::quote; Tok.bufEnd = curPos; return true;
      case '\'' : Tok.kind = tok::apostrophe; Tok.bufEnd = curPos; return true;
      case ','  : Tok.kind = tok::comma; Tok.bufEnd = curPos; return true;
      case 0    : LexEndOfFile(Tok); Tok.bufEnd = curPos -1; return false;
      }
    }
  }

  bool PunctuationLexer::LexEndOfFile(Token& Tok) {
    Tok.startToken();
    if (*curPos == '\0')
      Tok.kind = tok::eof;
    return Tok.kind != tok::unknown;
  }
} // end namespace cling
