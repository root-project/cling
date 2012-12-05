//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "MetaParser.h"

#include "MetaLexer.h"
#include "MetaSema.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InvocationOptions.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace cling {

  MetaParser::MetaParser(MetaSema* Actions) {
    m_Lexer.reset(0);
    m_Actions.reset(Actions);
    const InvocationOptions& Opts = Actions->getInterpreter().getOptions();
    MetaLexer metaSymbolLexer(Opts.MetaString);
    Token Tok;
    while(true) {
      metaSymbolLexer.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
      m_MetaSymbolCache.push_back(Tok);
    }
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
    while (Tok.isNot(stopAt) && Tok.isNot(tok::eof)){
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
    for (size_t i = 0; i < m_MetaSymbolCache.size(); ++i) {
      if (getCurTok().getKind() != m_MetaSymbolCache[i].getKind())
        return false;
      consumeToken();
    }
    return true;
  }

  bool MetaParser::isCommand() {
    return isLCommand() || isXCommand() || isqCommand() 
      || isUCommand() || isICommand() || israwInputCommand() 
      || isprintASTCommand() || isdynamicExtensionsCommand() || ishelpCommand()
      || isfileExCommand() || isfilesCommand() || isClassCommand() 
      || isgCommand() || isTypedefCommand();
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
        m_Actions->actOnLCommand(llvm::sys::Path(getCurTok().getIdent()));
        consumeToken();
        if (getCurTok().is(tok::comment)) {
          consumeAnyStringToken();
          m_Actions->actOnComment(getCurTok().getIdent());
        }
      }
    }
    // TODO: Some fine grained diagnostics
    return result;
  }

  // XCommand := 'x' FilePath[ArgList] | 'X' FilePath[ArgList]
  // FilePath := AnyString
  // ArgList := (ExtraArgList) ' ' [ArgList]
  // ExtraArgList := AnyString [, ExtraArgList]
  bool MetaParser::isXCommand() {
    bool result = false;
    const Token& Tok = getCurTok();
    if (Tok.is(tok::ident) && (Tok.getIdent().equals("x")
                               || Tok.getIdent().equals("X"))) {
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
      m_Actions->actOnxCommand(file, args);
      if (getCurTok().is(tok::comment)) {
        consumeAnyStringToken();
        m_Actions->actOnComment(getCurTok().getIdent());
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

  bool MetaParser::isqCommand() {
    bool result = false;
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("q")) {
      result = true;
      m_Actions->actOnqCommand();
    }
    return result;
  }

  bool MetaParser::isUCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("U")) {
      m_Actions->actOnUCommand();
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
      m_Actions->actOnICommand(path);
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
      m_Actions->actOnrawInputCommand(mode);
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
      m_Actions->actOnprintASTCommand(mode);
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
      m_Actions->actOndynamicExtensionsCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::ishelpCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("help")) {
      m_Actions->actOnhelpCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isfileExCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("fileEx")) {
      m_Actions->actOnfileExCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isfilesCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("files")) {
      m_Actions->actOnfilesCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isClassCommand() {
    const Token& Tok = getCurTok();
    if (Tok.is(tok::ident)) {
      if (Tok.getIdent().equals("class")) {
        consumeAnyStringToken(tok::eof);
        const Token& NextTok = getCurTok();
        llvm::StringRef className;
        if (NextTok.is(tok::raw_ident))
          className = NextTok.getIdent();
        m_Actions->actOnclassCommand(className);
        return true;
      }
      else if (Tok.getIdent().equals("Class")) {
        m_Actions->actOnClassCommand();
        return true;
      }
    }
    return false;
  }

  bool MetaParser::isgCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("g")) {
      consumeToken();
      SkipWhitespace();
      llvm::StringRef varName;
      if (getCurTok().is(tok::ident))
        varName = getCurTok().getIdent();
      m_Actions->actOngCommand(varName);
      return true;
    }
    return false;
  }
  
  bool MetaParser::isTypedefCommand() {
    const Token& Tok = getCurTok();
    if (Tok.is(tok::ident)) {
      if (Tok.getIdent().equals("typedef")) {
        consumeAnyStringToken();
        const Token& NextTok = getCurTok();
        llvm::StringRef typedefName;
        if (NextTok.is(tok::raw_ident))
          typedefName = NextTok.getIdent();
        m_Actions->actOnTypedefCommand(typedefName);
        return true;
      }
    }
    return false;
  }

} // end namespace cling
