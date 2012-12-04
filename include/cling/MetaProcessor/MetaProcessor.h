//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_METAPROCESSOR_H
#define CLING_METAPROCESSOR_H

#include "cling/Interpreter/Interpreter.h"

#include "clang/Lex/Token.h"

#include "llvm/ADT/OwningPtr.h"

#include <string>

namespace clang {
  class Lexer;
}

namespace cling {

  class Interpreter;
  class InputValidator;
  class MetaParser;
  class StoredValueRef;

  class MetaProcessorOpts {
  public:
    ///\brief is quit requested
    ///
    bool Quitting : 1;

    ///\brief is printAST requested
    ///
    bool PrintingAST : 1;

    ///\brief is using wrappers requested
    ///
    bool RawInput : 1;

    ///\brief is using dynamic scopes enabled
    ///
    bool DynamicLookup : 1;

    MetaProcessorOpts() {
      Quitting = 0;
      PrintingAST = 0;
      RawInput = 0;
      DynamicLookup = 0;
    }
  };

  ///\brief Class that helps processing meta commands, which add extra
  /// interactivity. Syntax .Command [arg0 arg1 ... argN]
  ///
  class MetaProcessor {
  private:
    ///\brief Reference to the interpreter
    ///
    Interpreter& m_Interp;

    ///\brief The input validator is used to figure out whether to switch to
    /// multiline mode or not. Checks for balanced parenthesis, etc.
    ///
    llvm::OwningPtr<InputValidator> m_InputValidator;

    ///\brief The parser used to parse our tiny "meta" language
    ///
    llvm::OwningPtr<MetaParser> m_MetaParser;

    ///\brief MetaProcessor's options
    ///
    MetaProcessorOpts m_Options;
     
    ///\brief Currently executing file as passed into executeFile
    ///
    llvm::StringRef m_CurrentlyExecutingFile;

    ///\brief Outermost currently executing file as passed into executeFile
    ///
    llvm::StringRef m_TopExecutingFile;
    
  private:

    ///\brief Handle one of the special commands in cling.
    /// Syntax .Command [arg0 arg1 ... argN]
    ///
    /// Known commands:
    /// @code .q @endcode - Quits
    /// @code .L <filename> @endcode - Loads the filename (It might be lib too)
    /// @code .(x|X) <filename>[(args)] @endcode - Loads the filename and
    /// executes the function with signature void filename(args)
    /// @code .printAST [0|1] @endcode - Toggles the printing of input's
    /// corresponding AST nodes.
    /// @code .rawInput [0|1] @endcode - Toggles wrapping and value printing of
    /// the input
    /// @code .I [path] @endcode - Dumps the include path. If argument is given
    /// adds the path to the list of include paths.
    /// @code .@ @endcode - Cancels multiline input
    /// @code .dynamicExtensions [0|1] @endcode - Toggles the use of the dynamic
    /// scopes and the late bining.
    /// @code .help @endcode - Show information about the usage of the commands
    /// @code .file @endcode - Show information about the loaded files
    ///
    ///\returns true if the command was known and thus handled.
    ///
    bool ProcessMeta(const std::string& input_line,
                     cling::StoredValueRef* result,
                     Interpreter::CompilationResult* compRes = 0);

  public:
    MetaProcessor(Interpreter& interp);
    ~MetaProcessor();

    MetaProcessorOpts& getMetaProcessorOpts();

    ///\brief Process the input coming from the prompt and possibli returns
    /// result of the execution of the last statement
    /// @param[in] input_line - the user input
    /// @param[out] result - the cling::StoredValueRef as result of the
    ///             execution of the last statement
    /// @param[out] compRes - whether compilation was successful
    ///
    ///\returns 0 on success or the indentation of the next input line should
    /// have in case of multi input mode.
    ///
    int process(const char* input_line, cling::StoredValueRef* result = 0,
                Interpreter::CompilationResult* compRes = 0);

    ///\brief When continuation is requested, this cancels and ignores previous
    /// input, resetting the continuation to a new line.
    void cancelContinuation();

    ///\brief Executes a file given the CINT specific rules. Mainly used as:
    /// .x filename[(args)], which in turn includes the filename and runs a
    /// function with signature void filename(args)
    /// @param[in] file - the filename
    /// @param[in] args - the args without ()
    /// @param[out] result - the cling::StoredValueRef as result of the
    ///             execution of the last statement
    /// @param[out] compRes - whether compilation was successful
    ///
    ///\returns true on success
    ///
    bool executeFile(llvm::StringRef file, llvm::StringRef args, 
                     cling::StoredValueRef* result = 0,
                     Interpreter::CompilationResult* compRes = 0);

    ///\brief Get the file name that is currently executing as passed to
    /// the currently active executeFile(). The returned StringRef::data() is
    /// NULL if no file is currently processed. For recursive calls to
    /// executeFile(), getCurrentlyExecutingFile() will return the nested file
    /// whereas getTopExecutingFile() returns the outer most file.
    llvm::StringRef getCurrentlyExecutingFile() const {
      return m_CurrentlyExecutingFile;
    }

    ///\brief Get the file name that is passed to the top most currently active
    /// executeFile(). The returned StringRef::data() is NULL if no file is
    /// currently processed.
    llvm::StringRef getTopExecutingFile() const {
      return m_TopExecutingFile;
    }
    
    ///\brief Reads prompt input from file.
    ///
    ///\param [in] filename - The file to read.
    /// @param[out] result - the cling::StoredValueRef as result of the
    ///             execution of the last statement
    ///\param [in] ignoreOutmostBlock - Whether to ignore enlosing {}.
    ///
    ///\returns result of the compilation.
    ///
    Interpreter::CompilationResult
    readInputFromFile(llvm::StringRef filename,
                      StoredValueRef* result /* = 0 */,
                      bool ignoreOutmostBlock = false);

  };
} // end namespace cling

#endif // CLING_METAPROCESSOR_H
