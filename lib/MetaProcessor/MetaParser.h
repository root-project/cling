//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_META_PARSER_H
#define CLING_META_PARSER_H

#include "MetaLexer.h" // for cling::Token

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

  // Command syntax: MetaCommand := <CommandSymbol><Command>
  //                 CommandSymbol := '.' | '//.'
  //                 Command := LCommand | xCommand | XCommand | qCommand |
  //                            UCommand | ICommand | RawInputCommand | 
  //                            PrintASTCommand | DynamicExtensionsCommand |
  //                            HelpCommand | FileExCommand | FilesCommand
  //                 LCommand := 'L' FilePath
  //                 qCommand := 'q'
  //                 xCommand := 'x' FilePath[ArgList]
  //                 XCommand := 'x' FilePath[ArgList]
  //                 UCommand := 'U'
  //                 ICommand := 'I' [FilePath]
  //                 RawInputCommand := 'rawInput' [Constant]
  //                 PrintASTCommand := 'printAST' [Constant]
  //                 DynamicExtensionsCommand := 'dynamicExtensions' [Constant]
  //                 DynamicExtensionsCommand := 'help'
  //                 DynamicExtensionsCommand := 'fileEx'
  //                 DynamicExtensionsCommand := 'files'
  //                 FilePath := AnyString
  //                 ArgList := (ExtraArgList) ' ' [ArgList]
  //                 ExtraArgList := AnyString [, ExtraArgList]
  //                 AnyString := *^(' ' | '\t')
  //                 Constant := 0|1
  class MetaParser {
  private:
    llvm::OwningPtr<MetaLexer> m_Lexer;
    llvm::OwningPtr<MetaSema> m_Actions;
    llvm::SmallVector<Token, 2> m_TokenCache;
  private:
    inline const Token& getCurTok() { return lookAhead(0); }
    void consumeToken();
    void consumeAnyStringToken(tok::TokenKind stopAt = tok::space);
    const Token& lookAhead(unsigned Num);
    void SkipWhitespace();

    bool isCommandSymbol();
    bool isCommand();
    bool isLCommand();
    bool isxCommand();
    bool isExtraArgList();
    bool isXCommand();
    bool isqCommand();
    bool isUCommand();
    bool isICommand();
    bool israwInputCommand();
    bool isprintASTCommand();
    bool isdynamicExtensionsCommand();
    bool ishelpCommand();
    bool isfileExCommand();
    bool isfilesCommand();    
  public:
    MetaParser(MetaSema* Actions);
    void enterNewInputLine(llvm::StringRef Line);
    bool isMetaCommand();
  };
} // end namespace cling

#endif // CLING_META_PARSER_H
