//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_META_PARSER_H
#define CLING_META_PARSER_H

#include "cling/MetaProcessor/MetaLexer.h" // for cling::Token
#include "cling/MetaProcessor/MetaSema.h" // for ActionResult
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace llvm {
  class StringRef;
}

namespace cling {
  class MetaLexer;
  class MetaSema;
  class Value;

  /// Command syntax: MetaCommand := < CommandSymbol >< Command >
  ///                 CommandSymbol := '.' | '//.'
  ///                 Command := LCommand | XCommand | qCommand | UCommand |
  ///                            ICommand | OCommand | RawInputCommand |
  ///                            PrintDebugCommand | DynamicExtensionsCommand |
  ///                            HelpCommand | FileExCommand | FilesCommand |
  ///                            ClassCommand | GCommand | StoreStateCommand |
  ///                            CompareStateCommand | StatsCommand | undoCommand
  ///                 LCommand := 'L' [FilePath]
  ///                 TCommand := 'T' FilePath FilePath
  ///                 >Command := '>' FilePath
  ///                 qCommand := 'q'
  ///                 XCommand := 'x' FilePath[ArgList] | 'X' FilePath[ArgList]
  ///                 UCommand := 'U' FilePath
  ///                 ICommand := 'I' [FilePath]
  ///                 OCommand := 'O'[' ']Constant
  ///                 RawInputCommand := 'rawInput' [Constant]
  ///                 PrintDebugCommand := 'printDebug' [Constant]
  ///                 DebugCommand := 'debug' [Constant]
  ///                 StoreStateCommand := 'storeState' "Ident"
  ///                 CompareStateCommand := 'compareState' "Ident"
  ///                 StatsCommand := 'stats' ['ast']
  ///                 traceCommand := 'trace' ['ast'] ["Ident"]
  ///                 undoCommand := 'undo' [Constant]
  ///                 DynamicExtensionsCommand := 'dynamicExtensions' [Constant]
  ///                 HelpCommand := 'help'
  ///                 FileExCommand := 'fileEx'
  ///                 FilesCommand := 'files'
  ///                 ClassCommand := 'class' AnyString | 'Class'
  ///                 GCommand := 'g' [Ident]
  ///                 FilePath := AnyString
  ///                 ArgList := (ExtraArgList) ' ' [ArgList]
  ///                 ExtraArgList := AnyString [, ExtraArgList]
  ///                 AnyString := *^(' ' | '\t')
  ///                 Constant := {0-9}
  ///                 Ident := a-zA-Z{a-zA-Z0-9}
  ///
  class MetaParser {
  private:
    MetaLexer m_Lexer;
    MetaSema &m_Actions;
    llvm::SmallVector<Token, 2> m_TokenCache;
    llvm::SmallVector<Token, 4> m_MetaSymbolCache;
  private:
    ///\brief Returns the current token without consuming it.
    ///
    inline const Token& getCurTok() { return lookAhead(0); }

    ///\brief Consume the current 'peek' token.
    ///
    void consumeToken();
    void consumeAnyStringToken(tok::TokenKind stopAt = tok::space);
    const Token& lookAhead(unsigned Num);
    void skipWhitespace();

    bool isCommandSymbol();
    bool isCommand(MetaSema::ActionResult& actionResult,
                   Value* resultValue);
    bool isLCommand(MetaSema::ActionResult& actionResult);
    bool isTCommand(MetaSema::ActionResult& actionResult);
    bool isRedirectCommand(MetaSema::ActionResult& actionResult);
    bool isExtraArgList();
    bool isXCommand(MetaSema::ActionResult& actionResult,
                    Value* resultValue);
    bool isAtCommand();
    bool isqCommand();
    bool isUCommand(MetaSema::ActionResult& actionResult);
    bool isICommand();
    bool isOCommand(MetaSema::ActionResult& actionResult);
    bool israwInputCommand();
    bool isdebugCommand();
    bool isprintDebugCommand();
    bool isstoreStateCommand();
    bool iscompareStateCommand();
    bool isstatsCommand();
    bool istraceCommand();
    bool isundoCommand();
    bool isdynamicExtensionsCommand();
    bool ishelpCommand();
    bool isfileExCommand();
    bool isfilesCommand();
    bool isClassCommand();
    bool isNamespaceCommand();
    bool isgCommand();
    bool isTypedefCommand();
    bool isShellCommand(MetaSema::ActionResult& actionResult,
                        Value* resultValue);
  public:
    MetaParser(MetaSema &Actions, llvm::StringRef Line);

    ///\brief Drives the recursive descendent parsing.
    ///
    ///\returns true if it was meta command.
    ///
    bool isMetaCommand(MetaSema::ActionResult& actionResult,
                       Value* resultValue);

    ///\brief Returns whether quit was requested via .q command
    ///
    bool isQuitRequested() const;

    MetaSema& getActions() const { return m_Actions; }
  };
} // end namespace cling

#endif // CLING_META_PARSER_H
