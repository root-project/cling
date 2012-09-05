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

  MetaProcessor::MetaProcessor(Interpreter& interp) : m_Interp(interp) {
    m_InputValidator.reset(new InputValidator());
  }

  MetaProcessor::~MetaProcessor() {}

  int MetaProcessor::process(const char* input_text, Value* result /*=0*/) {
    if (!input_text) { // null pointer, nothing to do.
      return 0;
    }
    if (!input_text[0]) { // empty string, nothing to do.
      return m_InputValidator->getExpectedIndent();
    }
    std::string input_line(input_text);
    if (input_line == "\n") { // just a blank line, nothing to do.
      return 0;
    }
    //  Check for and handle meta commands.
    bool was_meta = false;
    std::string& metaString = m_Interp.getOptions().MetaString;
    std::string::size_type lenMetaString = metaString.length();
    if (input_line.length() > lenMetaString
        && !input_line.compare(0, lenMetaString, metaString)) {
      was_meta = ProcessMeta(input_line.c_str() + lenMetaString, result);
    }
    if (was_meta) {
      return 0;
    }

    // Check if the current statement is now complete. If not, return to
    // prompt for more.
    if (m_InputValidator->validate(input_line, m_Interp.getCI()->getLangOpts())
        == InputValidator::kIncomplete) {
      return m_InputValidator->getExpectedIndent();
    }

    //  We have a complete statement, compile and execute it.
    std::string input = m_InputValidator->getInput();
    m_InputValidator->reset();
    if (m_Options.RawInput)
      m_Interp.declare(input);
    else
      m_Interp.process(input, result);

    return 0;
  }

  MetaProcessorOpts& MetaProcessor::getMetaProcessorOpts() {
    // Take interpreter's state
    m_Options.PrintingAST = m_Interp.isPrintingAST();
    return m_Options;
  }

  bool MetaProcessor::ProcessMeta(const std::string& input_line, Value* result){

   llvm::MemoryBuffer* MB = llvm::MemoryBuffer::getMemBuffer(input_line);
   LangOptions LO;
   LO.C99 = 1;
   // necessary for the @ symbol
   LO.ObjC1 = 1;
   Lexer RawLexer(SourceLocation(), LO, MB->getBufferStart(),
                  MB->getBufferStart(), MB->getBufferEnd());
   Token Tok;

   // Read the command
   RawLexer.LexFromRawLexer(Tok);
   if (!Tok.isAnyIdentifier() && Tok.isNot(tok::at))
     return false;

   const std::string Command = GetRawTokenName(Tok);
   std::string Param;

   //  .q //Quits
   if (Command == "q") {
      m_Options.Quitting = true;
      return true;
   }
   //  .L <filename>   //  Load code fragment.
   else if (Command == "L") {
     // TODO: Additional checks on params
     bool success
       = m_Interp.loadFile(SanitizeArg(ReadToEndOfBuffer(RawLexer, MB)));
     if (!success) {
       llvm::errs() << "Load file failed.\n";
     }
     return true;
   }
   //  .(x|X) <filename> //  Execute function from file, function name is
   //                    //  filename without extension.
   else if ((Command == "x") || (Command == "X")) {
     // TODO: Additional checks on params
     llvm::sys::Path path(SanitizeArg(ReadToEndOfBuffer(RawLexer, MB)));

     if (!path.isValid())
       return false;

     bool success = executeFile(path.c_str(), result);
      if (!success) {
        llvm::errs()<< "Execute file failed.\n";
      }
      return true;
   }
   //  .printAST [0|1]  // Toggle the printing of the AST or if 1 or 0 is given
   //                   // enable or disable it.
   else if (Command == "printAST") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (Tok.isNot(tok::numeric_constant) && Tok.isNot(tok::eof))
       return false;

     if (Tok.is(tok::eof)) {
       // toggle:
       bool print = !m_Interp.isPrintingAST();
       m_Interp.enablePrintAST(print);
       llvm::outs()<< (print?"P":"Not p") << "rinting AST\n";
     } else {
       Param = GetRawTokenName(Tok);

       if (Param == "0")
         m_Interp.enablePrintAST(false);
       else
         m_Interp.enablePrintAST(true);
     }

     m_Options.PrintingAST = m_Interp.isPrintingAST();
     return true;
   }
   //  .rawInput [0|1]  // Toggle the raw input or if 1 or 0 is given enable
   //                   // or disable it.
   else if (Command == "rawInput") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (Tok.isNot(tok::numeric_constant) && Tok.isNot(tok::eof))
       return false;

     if (Tok.is(tok::eof)) {
       // toggle:
       m_Options.RawInput = !m_Options.RawInput;
       llvm::outs() << (m_Options.RawInput?"U":"Not u") << "sing raw input\n";
     } else {
       Param = GetRawTokenName(Tok);

       if (Param == "0")
         m_Options.RawInput = false;
       else
         m_Options.RawInput = true;
     }
     return true;
   }
   //
   //  .U <filename>
   //
   //  Unload code fragment.
   //
   if (Command == "U") {
     // llvm::sys::Path path(param);
     // if (path.isDynamicLibrary()) {
     //   std::cerr << "[i] Failure: cannot unload shared libraries yet!"
     //             << std::endl;
     // }
     m_Interp.unload();
     return true;
   }
   //
   //  Unrecognized command.
   //
   //fprintf(stderr, "Unrecognized command.\n");
   else if (Command == "I") {

     // Check for params
     llvm::sys::Path path(SanitizeArg(ReadToEndOfBuffer(RawLexer, MB)));

     if (path.isEmpty())
       m_Interp.DumpIncludePath();
     else {
       // TODO: Additional checks on params

       if (path.isValid())
         m_Interp.AddIncludePath(path.c_str());
       else
         return false;
     }
     return true;
   }
  // Cancel the multiline input that has been requested
   else if (Command == "@") {
     m_InputValidator->reset();
     return true;
   }
   // Enable/Disable DynamicExprTransformer
   else if (Command == "dynamicExtensions") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (Tok.isNot(tok::numeric_constant) && Tok.isNot(tok::eof))
       return false;

     if (Tok.is(tok::eof)) {
       // toggle:
       bool dynlookup = !m_Interp.isDynamicLookupEnabled();
       m_Interp.enableDynamicLookup(dynlookup);
       llvm::outs() << (dynlookup?"U":"Not u") <<"sing dynamic extensions\n";
     } else {
       Param = GetRawTokenName(Tok);

       if (Param == "0")
         m_Interp.enableDynamicLookup(false);
       else
         m_Interp.enableDynamicLookup(true);
     }

     return true;
   }
   // Print Help
   else if (Command == "help") {
     PrintCommandHelp();
     return true;
   }
   // Print the loaded files
   else if (Command == "file") {
     PrintFileStats();
     return true;
   }

   return false;
  }

  std::string MetaProcessor::GetRawTokenName(const Token& Tok) {

    assert(!Tok.needsCleaning() && "Not implemented yet");

    switch (Tok.getKind()) {
    default:
      assert("Unknown token");
      return "";
    case tok::at:
      return "@";
    case tok::l_paren:
      return "(";
    case tok::r_paren:
      return ")";
    case tok::period:
      return ".";
    case tok::slash:
      return "/";
    case tok::numeric_constant:
      return StringRef(Tok.getLiteralData(), Tok.getLength()).str();
    case tok::raw_identifier:
      return StringRef(Tok.getRawIdentifierData(), Tok.getLength()).str();
    }
  }

  llvm::StringRef MetaProcessor::ReadToEndOfBuffer(Lexer& RawLexer,
                                                   llvm::MemoryBuffer* MB) {
    const char* CurPtr = RawLexer.getBufferLocation();
    if (CurPtr == MB->getBufferEnd()) {
      // Already at end of the buffer, return just the zero byte at the end.
      return StringRef(CurPtr, 0);
    }
    Token TmpTok;
    RawLexer.getAndAdvanceChar(CurPtr, TmpTok);
    return StringRef(CurPtr, MB->getBufferSize()-(CurPtr-MB->getBufferStart()));
  }

  llvm::StringRef MetaProcessor::SanitizeArg(const std::string& Str) {
    if(Str.empty())
      return Str;

    size_t begins = Str.find_first_not_of(" \t\n");
    size_t ends = Str.find_last_not_of(" \t\n") + 1;

    if (begins == std::string::npos)
      ends = begins + 1;

    return llvm::StringRef(Str.c_str() + begins, ends - begins);
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
  bool MetaProcessor::executeFile(const std::string& fileWithArgs,
                                  Value* result) {
    // Look for start of parameters:

    typedef std::pair<llvm::StringRef,llvm::StringRef> StringRefPair;

    StringRefPair pairFileArgs = llvm::StringRef(fileWithArgs).split('(');
    if (pairFileArgs.second.empty()) {
      pairFileArgs.second = ")";
    }
    StringRefPair pairPathFile = pairFileArgs.first.rsplit('/');
    if (pairPathFile.second.empty()) {
       pairPathFile.second = pairPathFile.first;
    }
    StringRefPair pairFuncExt = pairPathFile.second.rsplit('.');

    Interpreter::CompilationResult interpRes
       = m_Interp.declare(std::string("#include \"")
                          + pairFileArgs.first.str()
                          + std::string("\""));

    if (interpRes != Interpreter::kFailure) {
       std::string expression = pairFuncExt.first.str()
          + "(" + pairFileArgs.second.str();
       interpRes = m_Interp.evaluate(expression, result);
    }

    return (interpRes != Interpreter::kFailure);
  }
} // end namespace cling

