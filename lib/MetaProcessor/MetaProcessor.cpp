//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/MetaProcessor.h"

#include "InputValidator.h"
#include "cling/Interpreter/Interpreter.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Path.h"

using namespace clang;

namespace cling {

  namespace tok {
    enum TokenKind {
      commandstart,
      ident,
      l_paren,
      r_paren,
      anystring,
      comma,
      booltrue,
      boolfalse,
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
      bufEnd = 0;
      kind = tok::unknown;
    }
  public:
    tok::TokenKind getKind() const { return kind; }
    size_t getLength() const { return bufEnd - bufStart; }
    const char* getBufStart() const { return bufStart; }

    friend class CommandLexer;
  };

  class CommandLexer {
  private:
    const char* bufferStart;
    const char* bufferEnd;
    const char* curPos;
    const InvocationOptions& m_Opts;
  public:
    CommandLexer(const char* bufStart, const char* bufEnd, 
                 const InvocationOptions& Opts) 
      : bufferStart(bufStart), bufferEnd(bufEnd), curPos(bufStart), m_Opts(Opts)
    { }

    void LexBlankSpace() {
      // TODO: React on EOF.
      while (*curPos == ' ' || *curPos == '\t')
        ++curPos;
    }

    bool LexCommandSymbol(Token& Result) {
      Result.startToken();
      LexBlankSpace();
      Result.bufStart = curPos;
      for (size_t i = 0; i < m_Opts.MetaString.size(); ++i, ++curPos)
        if (*curPos != m_Opts.MetaString[i]) {
          Result.bufEnd = Result.bufStart;
          return false;
        }

      Result.bufEnd = curPos;
      return true;
    }

    bool LexIdent(Token& Result) {
      Result.startToken();
      Result.bufStart = curPos;
      unsigned char C = *curPos;
      while ((C >= 'A' && C <= 'Z') || (C >= 'a' && C <= 'z'))
        C = *++curPos;
      Result.bufEnd = curPos;
      if (Result.getLength() > 0) { 
        Result.kind = tok::ident;
        return true;
      } 
      return false;
    }

    bool LexAnyString(Token& Result) {
      Result.startToken();
      LexBlankSpace();
      Result.bufStart = curPos;
      unsigned char C = *curPos;
      while (C != ' ' && C != '\n' && C != '\t' && 
             C != '\0' && C != '(' && C != ')') {
        if (C=='\"') {
          do {
            C = *++curPos;
            if (C=='\\' && ( *(curPos+1) == '"') ) {
              // Skip escaped "
              C = *++curPos; // the "
              C = *++curPos; // the next character after that
            }
          } while (C!='"' && C!='\0');
          if (C == '\0') break;
        } else if (C=='\'') {
           do {
              C = *++curPos;
              if (C=='\\' && ( *(curPos+1) == '\'') ) {
                // Skip escaped "
                C = *++curPos; // the "
                C = *++curPos; // the next character after that
              }
           } while (C!='\'' && C!='\0');
           if (C == '\0') break; // Don't iterate past the end.
        }
        C = *++curPos;
      }
       
      Result.bufEnd = curPos;

      if (Result.getLength() > 0) { 
        Result.kind = tok::anystring;
        return true;
      }
      return false;        
    }

    bool LexSpecialSymbol(Token& Result) {
      Result.startToken();
      LexBlankSpace();
      Result.bufStart = curPos;
      if (*curPos == '(') {
        Result.kind = tok::l_paren;
        ++curPos;
      }
      else if (*curPos == ')') {
        Result.kind = tok::r_paren;
        ++curPos;
      }
      else if (*curPos == ',') {
        Result.kind = tok::comma;
        ++curPos;
      }
      Result.bufEnd = curPos;
      
      return Result.kind != tok::unknown;
    }

    bool LexBool(Token& Result) {
      Result.startToken();
      LexBlankSpace();
      Result.bufStart = curPos;
      if (*curPos == '0') {
        Result.kind = tok::boolfalse;
        ++curPos;
      }
      else if (*curPos == '1') {
        Result.kind = tok::booltrue;
        ++curPos;
      }
      Result.bufEnd = curPos;
      
      return Result.kind != tok::unknown;
    }

  };

  MetaProcessor::MetaProcessor(Interpreter& interp) : m_Interp(interp) {
    m_InputValidator.reset(new InputValidator());
  }

  MetaProcessor::~MetaProcessor() {}

  int MetaProcessor::process(const char* input_text,
                             StoredValueRef* result /*=0*/,
                             Interpreter::CompilationResult* compRes /*=0*/ ) {
    int expectedIndent = m_InputValidator->getExpectedIndent();
    if (compRes) {
      if (expectedIndent) {
        *compRes = Interpreter::kMoreInputExpected;
      } else {
        *compRes = Interpreter::kSuccess;
      }
    }
    if (!input_text || !input_text[0]) {
      // nullptr / empty string, nothing to do.
      return expectedIndent;
    }
    std::string input_line(input_text);
    if (input_line == "\n") { // just a blank line, nothing to do.
      return expectedIndent;
    }
    //  Check for and handle meta commands.
    if (ProcessMeta(input_line, result, compRes)) {
      return expectedIndent;
    }

    // Check if the current statement is now complete. If not, return to
    // prompt for more.
    if (m_InputValidator->validate(input_line, m_Interp.getCI()->getLangOpts())
        == InputValidator::kIncomplete) {
      if (compRes) *compRes = Interpreter::kMoreInputExpected;
      return m_InputValidator->getExpectedIndent();
    }

    //  We have a complete statement, compile and execute it.
    std::string input = m_InputValidator->getInput();
    m_InputValidator->reset();
    Interpreter::CompilationResult compResLocal;
    if (m_Options.RawInput)
      compResLocal = m_Interp.declare(input);
    else
      compResLocal = m_Interp.process(input, result);
    if (compRes) *compRes = compResLocal;

    return 0;
  }

  MetaProcessorOpts& MetaProcessor::getMetaProcessorOpts() {
    // Take interpreter's state
    m_Options.PrintingAST = m_Interp.isPrintingAST();
    return m_Options;
  }

  // Command syntax: meta_command := <command_symbol><command>[arg_list]
  //                 command_symbol := '.' | '//.'
  //                 command := ident
  //                 arg_list := any_string[(extra_arg_list)] [' ' arg_list]
  //                 extra_arg_list := any_string [, extra_arg_list]
  //
  bool MetaProcessor::ProcessMeta(const std::string& input_line,
                                  StoredValueRef* result,
                                Interpreter::CompilationResult* compRes /*=0*/){

   llvm::OwningPtr<llvm::MemoryBuffer> MB;
   MB.reset(llvm::MemoryBuffer::getMemBuffer(input_line));
   Token Tok;

   CommandLexer CmdLexer(MB->getBufferStart(), MB->getBufferEnd(), 
                         m_Interp.getOptions());

   if (!CmdLexer.LexCommandSymbol(Tok)) {
     // No error because it may be line containing code
     return false;
   }

   if (!CmdLexer.LexIdent(Tok)) {
     llvm::errs() << "Command name token expected. Try .help\n";
     if (compRes) *compRes = Interpreter::kFailure;
     return false;
   }

   llvm::StringRef Command (Tok.getBufStart(), Tok.getLength());
   // Should be used for faster comparison if the command is only one char long.
   unsigned char CmdStartChar = *Tok.getBufStart();
   if (compRes) *compRes = Interpreter::kSuccess;

   // .q Exits the process.
   if (CmdStartChar == 'q') {
     m_Options.Quitting = true;
     return true;
   }
   else if (CmdStartChar == 'L') {
     if (!CmdLexer.LexAnyString(Tok)) {
       llvm::errs() <<  "Filename expected.\n";
       if (compRes) *compRes = Interpreter::kFailure;
       return false;
     }
     //TODO: Check if the file exists and is readable.
     if (!m_Interp.loadFile(llvm::StringRef(Tok.getBufStart(), Tok.getLength()))) {
       llvm::errs() << "Load file failed.\n";
       if (compRes) *compRes = Interpreter::kFailure;
     }

     return true;
   }
   else if (CmdStartChar == 'x' || CmdStartChar == 'X') {
     if (!CmdLexer.LexAnyString(Tok)) {
       llvm::errs() << "Filename expected.\n";
       if (compRes) *compRes = Interpreter::kFailure;
       return false;
     }
     llvm::sys::Path file(llvm::StringRef(Tok.getBufStart(), Tok.getLength()));
     llvm::StringRef args;
     // TODO: Check whether the file exists using the compilers header search.
     // if (!file.canRead()) {
     //   llvm::errs() << "File doesn't exist or not readable.\n";
     //   return false;       
     // }
     CmdLexer.LexSpecialSymbol(Tok);
     if (Tok.getKind() == tok::l_paren) {
       // Good enough for now.
       if (!CmdLexer.LexAnyString(Tok)) {
         llvm::errs() << "Argument list expected.\n";
         if (compRes) *compRes = Interpreter::kFailure;
         return false;
       }
       args = llvm::StringRef(Tok.getBufStart(), Tok.getLength());
       if (!CmdLexer.LexSpecialSymbol(Tok) && Tok.getKind() == tok::r_paren) {
         llvm::errs() << "Closing parenthesis expected.\n";
         if (compRes) *compRes = Interpreter::kFailure;
         return false;
       }
     }
     if (!executeFile(file.str(), args, result, compRes))
       llvm::errs() << "Execute file failed.\n";
     return true;     
   }
   else if (CmdStartChar == 'U') {
     // if (!CmdLexer.LexAnyString(Tok)) {
     //   llvm::errs() << "Filename expected.\n";
     //   return false;
     // }
     // llvm::sys::Path file(llvm::StringRef(Tok.bufStart, Tok.getLength()));
     // TODO: Check whether the file exists using the compilers header search.
     // if (!file.canRead()) {
     //   llvm::errs() << "File doesn't exist or not readable.\n";
     //   return false;       
     // } else
     // if (file.isDynamicLibrary()) {
     //   llvm::errs() << "cannot unload shared libraries yet!.\n";
     //   return false;
     // }
     // TODO: Later comes more fine-grained unloading. For now just:
     m_Interp.unload();
     return true;
   }
   else if (CmdStartChar == '@') {
     m_InputValidator->reset();
     return true;
   }
   else if (CmdStartChar == 'I') {
     if (CmdLexer.LexAnyString(Tok)) {
       llvm::sys::Path path(llvm::StringRef(Tok.getBufStart(), Tok.getLength()));
       // TODO: Check whether the file exists using the compilers header search.
       // if (!path.canRead()) {
       //   llvm::errs() << "Path doesn't exist or not readable.\n";
       //   return false;       
       // }
       m_Interp.AddIncludePath(path.str());
     }
     else {
       m_Interp.DumpIncludePath();
     }
     return true;
   }
   else if (Command.equals("printAST")) {
     if (!CmdLexer.LexBool(Tok)) {
       bool flag = !m_Interp.isPrintingAST();
       m_Interp.enablePrintAST(flag);
       llvm::outs() << (flag ? "P" : "Not p") << "rinting AST\n";
     }
     else {
       if (Tok.getKind() == tok::boolfalse)
         m_Interp.enablePrintAST(false);
       else if (Tok.getKind() == tok::booltrue)
         m_Interp.enablePrintAST(true);
       else {
         llvm::errs() << "Boolean value expected.\n";
         if (compRes) *compRes = Interpreter::kFailure;
         return false;
       }
     }
     //m_Options.PrintingAST = m_Interp.isPrintingAST(); ????
     return true;
   }
   else if (Command.equals("rawInput")) {
     if (!CmdLexer.LexBool(Tok)) {
       m_Options.RawInput = !m_Options.RawInput;
       llvm::outs() << (m_Options.RawInput ? "U" :"Not u") << "sing raw input\n";
     }
     else {
       if (Tok.getKind() == tok::boolfalse)
         m_Options.RawInput = false;
       else if (Tok.getKind() == tok::booltrue)
         m_Options.RawInput = true;
       else {
         llvm::errs() << "Boolean value expected.\n";
         if (compRes) *compRes = Interpreter::kFailure;
         return false;
       }
     }
     return true;
   }
   else if (Command.equals("dynamicExtensions")) {
     if (!CmdLexer.LexBool(Tok)) {
       bool flag = !m_Interp.isDynamicLookupEnabled();
       m_Interp.enableDynamicLookup(flag);
       llvm::outs() << (flag ? "U" : "Not u") << "sing dynamic extensions\n";
     }
     else {
       if (Tok.getKind() == tok::boolfalse)
         m_Interp.enableDynamicLookup(false);
       else if (Tok.getKind() == tok::booltrue)
         m_Interp.enableDynamicLookup(true);
       else {
         llvm::errs() << "Boolean value expected.\n";
         if (compRes) *compRes = Interpreter::kFailure;
         return false;
       }
     }
     return true;
   }
   else if (Command.equals("help")) {
     PrintCommandHelp();
     return true;
   }
   else if (Command.equals("file")) {
     PrintFileStats();
     return true;
   }

   if (compRes) *compRes = Interpreter::kFailure;
   return false;
  }

  void MetaProcessor::PrintCommandHelp() {
    std::string& metaString = m_Interp.getOptions().MetaString;
    llvm::outs() << "Cling meta commands usage\n";
    llvm::outs() << "Syntax: .Command [arg0 arg1 ... argN]\n";
    llvm::outs() << "\n";
    llvm::outs() << metaString << "q\t\t\t\t- Exit the program\n";
    llvm::outs() << metaString << "L <filename>\t\t\t - Load file or library\n";
    llvm::outs() << metaString << "(x|X) <filename>[args]\t\t- Same as .L and runs a ";
    llvm::outs() << "function with signature ";
	llvm::outs() << "\t\t\t\tret_type filename(args)\n";
    llvm::outs() << metaString << "I [path]\t\t\t- Shows the include path. If a path is ";
    llvm::outs() << "given - \n\t\t\t\tadds the path to the include paths\n";
    llvm::outs() << metaString << "@ \t\t\t\t- Cancels and ignores the multiline input\n";
    llvm::outs() << metaString << "rawInput [0|1]\t\t\t- Toggle wrapping and printing ";
    llvm::outs() << "the execution\n\t\t\t\tresults of the input\n";
    llvm::outs() << metaString << "dynamicExtensions [0|1]\t- Toggles the use of the ";
    llvm::outs() << "dynamic scopes and the \t\t\t\tlate binding\n";
    llvm::outs() << metaString << "printAST [0|1]\t\t\t- Toggles the printing of input's ";
    llvm::outs() << "corresponding \t\t\t\tAST nodes\n";
    llvm::outs() << metaString << "help\t\t\t\t- Shows this information\n";
  }

  void MetaProcessor::PrintFileStats() {
    const SourceManager& SM = m_Interp.getCI()->getSourceManager();
    SM.getFileManager().PrintStats();

    llvm::outs() << "\n***\n\n";

    for (SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
           E = SM.fileinfo_end(); I != E; ++I) {
      llvm::outs() << (*I).first->getName();
      llvm::outs() << "\n";
    }
  }

  // Run a file: .x file[(args)]
  bool MetaProcessor::executeFile(llvm::StringRef file, llvm::StringRef args,
                                  StoredValueRef* result,
                               Interpreter::CompilationResult* compRes /*=0*/) {
    // Look for start of parameters:
    typedef std::pair<llvm::StringRef,llvm::StringRef> StringRefPair;

    // StringRefPair pairFileArgs = llvm::StringRef(fileWithArgs).split('(');
    // if (pairFileArgs.second.empty()) {
    //   pairFileArgs.second = ")";
    // }

    StringRefPair pairPathFile = file.rsplit('/');
    if (pairPathFile.second.empty()) {
       pairPathFile.second = pairPathFile.first;
    }
    StringRefPair pairFuncExt = pairPathFile.second.rsplit('.');

    Interpreter::CompilationResult interpRes = m_Interp.loadFile(file);
    if (interpRes == Interpreter::kSuccess) {
      std::string expression = pairFuncExt.first.str() + "(" + args.str() + ")";
      if (result)
        interpRes = m_Interp.evaluate(expression, *result);
      else
        interpRes = m_Interp.execute(expression);
    }
    if (compRes) *compRes = interpRes;
    return (interpRes != Interpreter::kFailure);
  }
} // end namespace cling

