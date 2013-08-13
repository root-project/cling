//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_META_PARSER_H
#define CLING_META_PARSER_H

#include "MetaLexer.h" // for cling::Token
#include "MetaSema.h" // for ActionResult
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  class StringRef;
  namespace sys {
    class Path;
  }
}

namespace cling {
  class MetaLexer;
  class MetaSema;
  class StoredValueRef;

  // Command syntax: MetaCommand := <CommandSymbol><Command>
  //                 CommandSymbol := '.' | '//.'
  //                 Command := LCommand | XCommand | qCommand | UCommand |
  //                            ICommand | OCommand | RawInputCommand |
  //                            PrintASTCommand | DynamicExtensionsCommand | HelpCommand |
  //                            FileExCommand | FilesCommand | ClassCommand |
  //                            GCommand | PrintIRCommand
  //                 LCommand := 'L' FilePath
  //                 qCommand := 'q'
  //                 XCommand := 'x' FilePath[ArgList] | 'X' FilePath[ArgList]
  //                 UCommand := 'U'
  //                 ICommand := 'I' [FilePath]
  //                 OCommand := 'O'[' ']OptimizationLevel
  //                 RawInputCommand := 'rawInput' [Constant]
  //                 PrintASTCommand := 'printAST' [Constant]
  //                 PrintIRCommand := 'printIR' [Constant]
  //                 DynamicExtensionsCommand := 'dynamicExtensions' [Constant]
  //                 HelpCommand := 'help'
  //                 FileExCommand := 'fileEx'
  //                 FilesCommand := 'files'
  //                 ClassCommand := 'class' AnyString | 'Class'
  //                 GCommand := 'g' [Ident]
  //                 FilePath := AnyString
  //                 ArgList := (ExtraArgList) ' ' [ArgList]
  //                 ExtraArgList := AnyString [, ExtraArgList]
  //                 AnyString := *^(' ' | '\t')
  //                 Constant := 0|1
  //                 Ident := a-zA-Z{a-zA-Z0-9}
  //                 OptimizationLevel := OptimizationLevel{0-9}
  //
  class MetaParser {
  private:
    llvm::OwningPtr<MetaLexer> m_Lexer;
    llvm::OwningPtr<MetaSema> m_Actions;
    llvm::SmallVector<Token, 2> m_TokenCache;
    llvm::SmallVector<Token, 4> m_MetaSymbolCache;
  private:
    inline const Token& getCurTok() { return lookAhead(0); }
    void consumeToken();
    void consumeAnyStringToken(tok::TokenKind stopAt = tok::space);
    const Token& lookAhead(unsigned Num);
    void skipWhitespace();

    bool isCommandSymbol();
    bool isCommand(MetaSema::ActionResult& actionResult,
                   StoredValueRef* resultValue);
    bool isLCommand(MetaSema::ActionResult& actionResult);
    bool isExtraArgList();
    bool isXCommand(MetaSema::ActionResult& actionResult,
                    StoredValueRef* resultValue);
    bool isqCommand();
    bool isUCommand(MetaSema::ActionResult& actionResult);
    bool isICommand();
    bool isOCommand();
    bool israwInputCommand();
    bool isprintASTCommand();
    bool isprintIRCommand();
    bool isdynamicExtensionsCommand();
    bool ishelpCommand();
    bool isfileExCommand();
    bool isfilesCommand();
    bool isClassCommand();
    bool isgCommand();
    bool isTypedefCommand();
    bool isShellCommand(MetaSema::ActionResult& actionResult,
                        StoredValueRef* resultValue);
  public:
    MetaParser(MetaSema* Actions);
    void enterNewInputLine(llvm::StringRef Line);

    ///\brief Drives the recursive decendent parsing.
    ///
    ///\returns true if it was meta command.
    ///
    bool isMetaCommand(MetaSema::ActionResult& actionResult,
                       StoredValueRef* resultValue);

    ///\brief Returns whether quit was requested via .q command
    ///
    bool isQuitRequested() const;

    MetaSema& getActions() const { return *m_Actions.get(); }
  };
} // end namespace cling

#endif // CLING_META_PARSER_H
