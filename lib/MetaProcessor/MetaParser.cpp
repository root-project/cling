//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "MetaParser.h"

#include "MetaLexer.h"
#include "MetaSema.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace cling {

  MetaParser::MetaParser(MetaSema* Actions) {
    m_Lexer.reset(0);
    m_Actions.reset(Actions);
  }

  void MetaParser::enterNewInputLine(llvm::StringRef Line) {
    m_Lexer.reset(new MetaLexer(Line));
    m_TokenCache.clear();
  }

  void MetaParser::consumeToken() {
    if (m_TokenCache.size())
      m_TokenCache.erase(m_TokenCache.begin());
    
    lookAhead(0);
  }

  void MetaParser::consumeAnyStringToken(tok::TokenKind stopAt/*=tok::space*/) {
    consumeToken();
    // we have to merge the tokens from the queue until we reach eof token or
    // space token
    SkipWhitespace();
    // Add the new token in which we will merge the others.
    Token& MergedTok = m_TokenCache.front();

    if (MergedTok.is(stopAt) || MergedTok.is(tok::eof) 
        || MergedTok.is(tok::comment))
      return;

    Token Tok = lookAhead(1);
    while (Tok.isNot(stopAt) && Tok.isNot(tok::eof) && Tok.isNot(tok::comment)){
      //MergedTok.setLength(MergedTok.getLength() + Tok.getLength());
      m_TokenCache.erase(m_TokenCache.begin() + 1);
      Tok = lookAhead(1);
    }
    MergedTok.setKind(tok::raw_ident);
    MergedTok.setLength(Tok.getBufStart() - MergedTok.getBufStart());
  }

  const Token& MetaParser::lookAhead(unsigned N) {
    if (N < m_TokenCache.size())
      return m_TokenCache[N];

    for (unsigned C = N+1 - m_TokenCache.size(); C > 0; --C) {
      m_TokenCache.push_back(Token());
      m_Lexer->Lex(m_TokenCache.back());
    }
    return m_TokenCache.back();
  }

  void MetaParser::SkipWhitespace() {
    while(getCurTok().is(tok::space))
      consumeToken();
  }

  bool MetaParser::isMetaCommand() {
    return isCommandSymbol() && isCommand();
  }

  bool MetaParser::isCommandSymbol() {
    consumeToken();
    if (getCurTok().is(tok::dot) /*TODO: || Tok.is(//.)*/)
      return true;
    return false;
  }

  bool MetaParser::isCommand() {
    consumeToken();
    return isLCommand() || isxCommand() || isXCommand() || isqCommand() 
      || isUCommand() || isICommand() || israwInputCommand() 
      || isprintASTCommand() || isdynamicExtensionsCommand() || ishelpCommand()
      || isfileExCommand() || isfilesCommand();
  }

  // L := 'L' FilePath
  // FilePath := AnyString
  // AnyString := .*^(' ' | '\t')
  bool MetaParser::isLCommand() {
    bool result = false;
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("L")) {
      consumeAnyStringToken();
      if (getCurTok().is(tok::raw_ident)) {
        result = true;
        m_Actions->ActOnLCommand(llvm::sys::Path(getCurTok().getIdent()));
        consumeToken();
        if (getCurTok().is(tok::comment)) {
          consumeAnyStringToken();
          m_Actions->ActOnComment(getCurTok().getIdent());
        }
      }
    }
    // TODO: Some fine grained diagnostics
    return result;
  }

  // xCommand := 'x' FilePath[ArgList]
  // FilePath := AnyString
  // ArgList := (ExtraArgList) ' ' [ArgList]
  // ExtraArgList := AnyString [, ExtraArgList]
  bool MetaParser::isxCommand() {
    bool result = false;
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("x")) {
      // There might be ArgList
      consumeAnyStringToken(tok::l_paren);
      llvm::sys::Path file(getCurTok().getIdent());
      llvm::StringRef args;
      result = true;
      consumeToken();
      if (getCurTok().is(tok::l_paren) && isExtraArgList()) {
        args = getCurTok().getIdent();
        consumeToken(); // consume the closing paren
      }
      m_Actions->ActOnxCommand(file, args);
      if (getCurTok().is(tok::comment)) {
        consumeAnyStringToken();
        m_Actions->ActOnComment(getCurTok().getIdent());
      }

    }

    return result;
  }

  // ExtraArgList := AnyString [, ExtraArgList]
  bool MetaParser::isExtraArgList() {
    // This might be expanded if we need better arg parsing.
    consumeAnyStringToken(tok::r_paren);
    
    return getCurTok().is(tok::raw_ident);
  }

  bool MetaParser::isXCommand() {
    // TODO: For now we don't distinguish both cases. In future we will have to.
    return isxCommand();
  }

  bool MetaParser::isqCommand() {
    bool result = false;
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("q")) {
      result = true;
      m_Actions->ActOnqCommand();
    }
    return result;
  }

  bool MetaParser::isUCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("q")) {
      m_Actions->ActOnUCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isICommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("I")) {
      consumeAnyStringToken();
      llvm::sys::Path path;
      if (getCurTok().is(tok::raw_ident))
        path = getCurTok().getIdent();
      m_Actions->ActOnICommand(path);
      return true;
    }
    return false;
  }

  bool MetaParser::israwInputCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("rawInput")) {
      MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      SkipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = (MetaSema::SwitchMode)getCurTok().getConstant();
      m_Actions->ActOnrawInputCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::isprintASTCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("printAST")) {
      MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      SkipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = (MetaSema::SwitchMode)getCurTok().getConstant();
      m_Actions->ActOnprintASTCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::isdynamicExtensionsCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("dynamicExtensions")) {
      MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      SkipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = (MetaSema::SwitchMode)getCurTok().getConstant();
      m_Actions->ActOndynamicExtensionsCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::ishelpCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("help")) {
      m_Actions->ActOnhelpCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isfileExCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("fileEx")) {
      m_Actions->ActOnfileExCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isfilesCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("files")) {
      m_Actions->ActOnfilesCommand();
      return true;
    }
    return false;
  }

} // end namespace cling
