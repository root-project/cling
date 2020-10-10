//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/MetaParser.h"
#include "cling/MetaProcessor/MetaSema.h"
#include "cling/MetaProcessor/MetaLexer.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Interpreter/Value.h"

#include "cling/Utils/Output.h"
#include "cling/Utils/Paths.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace cling {

  MetaParser::MetaParser(MetaSema &Actions, llvm::StringRef Line) :
  m_Lexer(Line), m_Actions(Actions) {
    const InvocationOptions& Opts = Actions.getInterpreter().getOptions();
    MetaLexer metaSymbolLexer(Opts.MetaString);
    Token Tok;
    while(true) {
      metaSymbolLexer.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
      m_MetaSymbolCache.push_back(Tok);
    }
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
    skipWhitespace();
    // Add the new token in which we will merge the others.
    Token& MergedTok = m_TokenCache.front();

    if (MergedTok.is(stopAt) || MergedTok.is(tok::eof)
        || MergedTok.is(tok::comment))
      return;

    //look ahead for the next token without consuming it
    Token Tok = lookAhead(1);
    Token PrevTok = Tok;
    while (Tok.isNot(stopAt) && Tok.isNot(tok::eof)){
      //MergedTok.setLength(MergedTok.getLength() + Tok.getLength());
      m_TokenCache.erase(m_TokenCache.begin() + 1);
      PrevTok = Tok;
      //look ahead for the next token without consuming it
      Tok = lookAhead(1);
    }
    MergedTok.setKind(tok::raw_ident);
    if (PrevTok.is(tok::space)) {
      // for "id <space> eof" the merged token should contain "id", not
      // "id <space>".
      Tok = PrevTok;
    }
    MergedTok.setLength(Tok.getBufStart() - MergedTok.getBufStart());
  }

  const Token& MetaParser::lookAhead(unsigned N) {
    if (N < m_TokenCache.size())
      return m_TokenCache[N];

    for (unsigned C = N+1 - m_TokenCache.size(); C > 0; --C) {
      m_TokenCache.push_back(Token());
      m_Lexer.Lex(m_TokenCache.back());
    }
    return m_TokenCache.back();
  }

  void MetaParser::skipWhitespace() {
    while(getCurTok().is(tok::space))
      consumeToken();
  }

  bool MetaParser::isMetaCommand(MetaSema::ActionResult& actionResult,
                                 Value* resultValue) {
    return isCommandSymbol() && isCommand(actionResult, resultValue);
  }

  bool MetaParser::isQuitRequested() const {
    return m_Actions.isQuitRequested();
  }

  bool MetaParser::isCommandSymbol() {
    for (size_t i = 0; i < m_MetaSymbolCache.size(); ++i) {
      if (getCurTok().getKind() != m_MetaSymbolCache[i].getKind())
        return false;
      consumeToken();
    }
    return true;
  }

  bool MetaParser::isCommand(MetaSema::ActionResult& actionResult,
                             Value* resultValue) {
    if (resultValue)
      *resultValue = Value();
    // Assume success; some actions don't set it.
    actionResult = MetaSema::AR_Success;
    return isLCommand(actionResult)
      || isXCommand(actionResult, resultValue) ||isTCommand(actionResult)
      || isAtCommand()
      || isqCommand() || isUCommand(actionResult) || isICommand()
      || isOCommand(actionResult) || israwInputCommand()
      || isdebugCommand() || isprintDebugCommand()
      || isdynamicExtensionsCommand() || ishelpCommand() || isfileExCommand()
      || isfilesCommand() || isClassCommand() || isNamespaceCommand() || isgCommand()
      || isTypedefCommand()
      || isShellCommand(actionResult, resultValue) || isstoreStateCommand()
      || iscompareStateCommand() || isstatsCommand() || isundoCommand()
      || isRedirectCommand(actionResult) || istraceCommand();
  }

  // L := 'L' FilePath Comment
  // FilePath := AnyString
  // AnyString := .*^('\t' Comment)
  bool MetaParser::isLCommand(MetaSema::ActionResult& actionResult) {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("L")) {
      consumeAnyStringToken(tok::comment);
      llvm::StringRef filePath;
      if (getCurTok().is(tok::raw_ident)) {
        filePath = getCurTok().getIdent();
        consumeToken();
        if (getCurTok().is(tok::comment)) {
          consumeAnyStringToken(tok::eof);
          m_Actions.actOnComment(getCurTok().getIdent());
        }
      }
      actionResult = m_Actions.actOnLCommand(filePath);
      return true;
    }
    // TODO: Some fine grained diagnostics
    return false;
  }

  // T := 'T' FilePath Comment
  // FilePath := AnyString
  // AnyString := .*^('\t' Comment)
  bool MetaParser::isTCommand(MetaSema::ActionResult& actionResult) {
    bool result = false;
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("T")) {
      consumeAnyStringToken();
      if (getCurTok().is(tok::raw_ident)) {
        std::string inputFile = getCurTok().getIdent();
        consumeAnyStringToken(tok::eof);
        if (getCurTok().is(tok::raw_ident)) {
          result = true;
          std::string outputFile = getCurTok().getIdent();
          actionResult = m_Actions.actOnTCommand(inputFile, outputFile);
        }
      }
    }
    // TODO: Some fine grained diagnostics
    return result;
  }

  // >RedirectCommand := '>' FilePath
  // FilePath := AnyString
  // AnyString := .*^(' ' | '\t')
  bool MetaParser::isRedirectCommand(MetaSema::ActionResult& actionResult) {

    unsigned constant_FD = 0;
    // Default redirect is stdout.
    MetaProcessor::RedirectionScope stream = MetaProcessor::kSTDOUT;

    if (getCurTok().is(tok::constant) && lookAhead(1).is(tok::greater)) {
      // > or 1> the redirection is for stdout stream
      // 2> redirection for stderr stream
      constant_FD = getCurTok().getConstant();
      if (constant_FD == 2) {
        stream = MetaProcessor::kSTDERR;
      // Wrong constant_FD, do not redirect.
      } else if (constant_FD != 1) {
        cling::errs() << "cling::MetaParser::isRedirectCommand():"
                      << "invalid file descriptor number " << constant_FD <<"\n";
        return true;
      }
      consumeToken();
    }
    // &> redirection for both stdout & stderr
    if (getCurTok().is(tok::ampersand)) {
      if (constant_FD == 0) {
        stream = MetaProcessor::kSTDBOTH;
      }
      consumeToken();
    }
    llvm::StringRef file;
    if (getCurTok().is(tok::greater)) {
      bool append = false;
      // check whether we have >>
      if (lookAhead(1).is(tok::greater)) {
        consumeToken();
        append = true;
      }
      // check for syntax like: 2>&1
      if (lookAhead(1).is(tok::ampersand)) {
        if (constant_FD == 0)
          stream = MetaProcessor::kSTDBOTH;

        const Token& Tok = lookAhead(2);
        if (Tok.is(tok::constant)) {
          switch (Tok.getConstant()) {
            case 1: file = llvm::StringRef("&1"); break;
            case 2: file = llvm::StringRef("&2"); break;
            default: break;
          }
          if (!file.empty()) {
            // Mark the stream name as refering to stderr or stdout, not a name
            stream = MetaProcessor::RedirectionScope(stream |
                                                     MetaProcessor::kSTDSTRM);
            consumeToken(); // &
            consumeToken(); // 1,2
          }
        }
      }
      std::string EnvExpand;
      if (!lookAhead(1).is(tok::eof) && !(stream & MetaProcessor::kSTDSTRM)) {
        consumeAnyStringToken(tok::eof);
        if (getCurTok().is(tok::raw_ident)) {
          EnvExpand = getCurTok().getIdent();
          // Quoted path, no expansion and strip quotes
          if (EnvExpand.size() > 3 && EnvExpand.front() == '"' &&
              EnvExpand.back() == '"') {
            file = EnvExpand;
            file = file.substr(1, file.size()-2);
          } else if (!EnvExpand.empty()) {
            cling::utils::ExpandEnvVars(EnvExpand);
            file = EnvExpand;
          }
          consumeToken();
          // If we had a token, we need a path; empty means to undo a redirect
          if (file.empty())
            return false;
        }
      }
      // Empty file means std.
      actionResult =
          m_Actions.actOnRedirectCommand(file/*file*/,
                                         stream/*which stream to redirect*/,
                                         append/*append mode*/);
      return true;
    }
    return false;
  }

  // XCommand := 'x' FilePath[ArgList] | 'X' FilePath[ArgList]
  // FilePath := AnyString
  // ArgList := (ExtraArgList) ' ' [ArgList]
  // ExtraArgList := AnyString [, ExtraArgList]
  bool MetaParser::isXCommand(MetaSema::ActionResult& actionResult,
                              Value* resultValue) {
    if (resultValue)
      *resultValue = Value();
    const Token& Tok = getCurTok();
    if (Tok.is(tok::ident) && (Tok.getIdent().equals("x")
                               || Tok.getIdent().equals("X"))) {
      consumeToken();
      skipWhitespace();

      // There might be an ArgList:
      int forward = 0;
      std::string args;
      llvm::StringRef file(getCurTok().getBufStart());
      while (!lookAhead(forward).is(tok::eof))
	++forward;

      // Skip any trailing ';':
      if (lookAhead(forward - 1).is(tok::semicolon))
	--forward;

      // Now track back to find the opening '('.
      if (lookAhead(forward - 1).is(tok::r_paren)) {
	// Trailing ')' - we interpret that as an argument.
	--forward; // skip ')'
	int nesting = 1;
	while (--forward > 0 && nesting) {
	  if (lookAhead(forward).is(tok::l_paren))
	    --nesting;
	  else if (lookAhead(forward).is(tok::r_paren))
	    ++nesting;
	}
	if (forward == 0) {
	  cling::errs() << "cling::MetaParser::isXCommand():"
	    "error parsing argument in " << getCurTok().getBufStart() << '\n';
	  // interpret everything as "the file"
	} else {
	  while (forward--)
	    consumeToken();
	  consumeToken(); // the forward-0 token.
	  args = getCurTok().getBufStart();
	  file = file.drop_back(args.length());
	}
      }

      if (args.empty())
        args = "()";
      actionResult = m_Actions.actOnxCommand(file, args, resultValue);
      return true;
    }

    return false;
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
      m_Actions.actOnqCommand();
    }
    return result;
  }

  bool MetaParser::isUCommand(MetaSema::ActionResult& actionResult) {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("U")) {
      consumeAnyStringToken(tok::eof);
      llvm::StringRef path;
      if (getCurTok().is(tok::raw_ident)) {
        path = getCurTok().getIdent();
        actionResult = m_Actions.actOnUCommand(path);
        return true;
      }
    }
    return false;
  }

  bool MetaParser::isICommand() {
    if (getCurTok().is(tok::ident) &&
        (   getCurTok().getIdent().equals("I")
         || getCurTok().getIdent().equals("include"))) {
      consumeAnyStringToken(tok::eof);
      llvm::StringRef path;
      if (getCurTok().is(tok::raw_ident))
        path = getCurTok().getIdent();
      m_Actions.actOnICommand(path);
      return true;
    }
    return false;
  }

  bool MetaParser::isOCommand(MetaSema::ActionResult& actionResult) {
    const Token& currTok = getCurTok();
    if (currTok.is(tok::ident)) {
      llvm::StringRef ident = currTok.getIdent();
      if (ident.startswith("O")) {
        if (ident.size() > 1) {
          int level = 0;
          if (!ident.substr(1).getAsInteger(10, level) && level >= 0) {
            consumeAnyStringToken(tok::eof);
            if (getCurTok().is(tok::raw_ident))
              return false;
            actionResult = m_Actions.actOnOCommand(level);
            return true;
          }
        } else {
          consumeAnyStringToken(tok::eof);
          const Token& lastStringToken = getCurTok();
          if (lastStringToken.is(tok::raw_ident)
              && lastStringToken.getLength()) {
            int level = 0;
            if (!lastStringToken.getIdent().getAsInteger(10, level)
                && level >= 0) {
              actionResult = m_Actions.actOnOCommand(level);
              return true;
            }
          } else {
            m_Actions.actOnOCommand();
            actionResult = MetaSema::AR_Success;
            return true;
          }
        }
      }
    }

    return false;
  }

  bool MetaParser::isAtCommand() {
    if (getCurTok().is(tok::at) // && getCurTok().getIdent().equals("@")
        ) {
      consumeToken();
      skipWhitespace();
      m_Actions.actOnAtCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::israwInputCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("rawInput")) {
      MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      skipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = (MetaSema::SwitchMode)getCurTok().getConstantAsBool();
      m_Actions.actOnrawInputCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::isdebugCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("debug")) {
      llvm::Optional<int> mode;
      consumeToken();
      skipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = getCurTok().getConstant();
      m_Actions.actOndebugCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::isprintDebugCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("printDebug")) {
      MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      skipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = (MetaSema::SwitchMode)getCurTok().getConstantAsBool();
      m_Actions.actOnprintDebugCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::isstoreStateCommand() {
     if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("storeState")) {
       //MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      skipWhitespace();
      if (!getCurTok().is(tok::stringlit))
        return false; // FIXME: Issue proper diagnostics
      std::string ident = getCurTok().getIdentNoQuotes();
      consumeToken();
      m_Actions.actOnstoreStateCommand(ident);
      return true;
    }
    return false;
  }

  bool MetaParser::iscompareStateCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("compareState")) {
      //MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      skipWhitespace();
      if (!getCurTok().is(tok::stringlit))
        return false; // FIXME: Issue proper diagnostics
      std::string ident = getCurTok().getIdentNoQuotes();
      consumeToken();
      m_Actions.actOncompareStateCommand(ident);
      return true;
    }
    return false;
  }

  bool MetaParser::isstatsCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("stats")) {
      consumeToken();
      skipWhitespace();
      if (!getCurTok().is(tok::ident))
        return false; // FIXME: Issue proper diagnostics
      llvm::StringRef what = getCurTok().getIdent();
      consumeToken();
      skipWhitespace();
      const Token& next = getCurTok();
      m_Actions.actOnstatsCommand(what, next.is(tok::ident)
                                         ? next.getIdent() : llvm::StringRef());
      return true;
    }
    return false;
  }

  // dumps/creates a trace of the requested representation.
  bool MetaParser::istraceCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("trace")) {
      consumeToken();
      skipWhitespace();
      if (!getCurTok().is(tok::ident))
          return false;
      llvm::StringRef ident = getCurTok().getIdent();
      consumeToken();
      skipWhitespace();
      m_Actions.actOnstatsCommand(ident.equals("ast")
        ? llvm::StringRef("asttree") : ident,
        getCurTok().is(tok::ident) ? getCurTok().getIdent() : llvm::StringRef());
      consumeToken();
      return true;
    }
    return false;
  }

  bool MetaParser::isundoCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("undo")) {
      consumeToken();
      skipWhitespace();
      const Token& next = getCurTok();
      if (next.is(tok::constant))
        m_Actions.actOnUndoCommand(next.getConstant());
      else
        m_Actions.actOnUndoCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isdynamicExtensionsCommand() {
    if (getCurTok().is(tok::ident) &&
        getCurTok().getIdent().equals("dynamicExtensions")) {
      MetaSema::SwitchMode mode = MetaSema::kToggle;
      consumeToken();
      skipWhitespace();
      if (getCurTok().is(tok::constant))
        mode = (MetaSema::SwitchMode)getCurTok().getConstantAsBool();
      m_Actions.actOndynamicExtensionsCommand(mode);
      return true;
    }
    return false;
  }

  bool MetaParser::ishelpCommand() {
    const Token& Tok = getCurTok();
    if (Tok.is(tok::quest_mark) ||
        (Tok.is(tok::ident) && Tok.getIdent().equals("help"))) {
      m_Actions.actOnhelpCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isfileExCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("fileEx")) {
      m_Actions.actOnfileExCommand();
      return true;
    }
    return false;
  }

  bool MetaParser::isfilesCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("files")) {
      m_Actions.actOnfilesCommand();
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
        m_Actions.actOnclassCommand(className);
        return true;
      }
      else if (Tok.getIdent().equals("Class")) {
        m_Actions.actOnClassCommand();
        return true;
      }
    }
    return false;
  }

  bool MetaParser::isNamespaceCommand() {
    const Token& Tok = getCurTok();
    if (Tok.is(tok::ident)) {
      if (Tok.getIdent().equals("namespace")) {
        consumeAnyStringToken(tok::eof);
        if (getCurTok().is(tok::raw_ident))
          return false;
        m_Actions.actOnNamespaceCommand();
        return true;
      }
    }
    return false;
  }

  bool MetaParser::isgCommand() {
    if (getCurTok().is(tok::ident) && getCurTok().getIdent().equals("g")) {
      consumeToken();
      skipWhitespace();
      llvm::StringRef varName;
      if (getCurTok().is(tok::ident))
        varName = getCurTok().getIdent();
      m_Actions.actOngCommand(varName);
      return true;
    }
    return false;
  }

  bool MetaParser::isTypedefCommand() {
    const Token& Tok = getCurTok();
    if (Tok.is(tok::ident)) {
      if (Tok.getIdent().equals("typedef")) {
        consumeAnyStringToken(tok::eof);
        const Token& NextTok = getCurTok();
        llvm::StringRef typedefName;
        if (NextTok.is(tok::raw_ident))
          typedefName = NextTok.getIdent();
        m_Actions.actOnTypedefCommand(typedefName);
        return true;
      }
    }
    return false;
  }

  bool MetaParser::isShellCommand(MetaSema::ActionResult& actionResult,
                                  Value* resultValue) {
    if (resultValue)
      *resultValue = Value();
    const Token& Tok = getCurTok();
    if (Tok.is(tok::excl_mark)) {
      consumeAnyStringToken(tok::eof);
      const Token& NextTok = getCurTok();
      if (NextTok.is(tok::raw_ident)) {
         llvm::StringRef commandLine(NextTok.getIdent());
         if (!commandLine.empty())
            actionResult = m_Actions.actOnShellCommand(commandLine,
                                                        resultValue);
      }
      return true;
    }
    return false;
  }

} // end namespace cling
