//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Lukasz Janyst <ljanyst@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_METAPROCESSOR_H
#define CLING_METAPROCESSOR_H

#include "cling/Interpreter/Interpreter.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"

namespace cling {

  class Interpreter;
  class InputValidator;
  class MetaParser;
  class StoredValueRef;

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

    ///\brief Currently executing file as passed into executeFile
    ///
    llvm::StringRef m_CurrentlyExecutingFile;

    ///\brief Outermost currently executing file as passed into executeFile
    ///
    llvm::StringRef m_TopExecutingFile;

    ///\brief The output stream being used for various purposes.
    ///
    llvm::raw_ostream& m_Outs;

    //brief The file name for the redirection of the stdout.
    ///
    std::string m_FileOut;

    ///brief The file name for the redirection of the stderr.
    ///
    std::string m_FileErr;

  public:
    enum RedirectionScope {
      kSTDOUT = 1,
      kSTDERR = 2,
      kSTDBOTH = 3
    };

  public:
    ///brief Class to be created for each processing input to be
    /// able to redirect std.
    class MaybeRedirectOutputRAII {
    private:
      MetaProcessor* m_MetaProcessor;
      char *terminalOut;
      char *terminalErr;

    public:
      MaybeRedirectOutputRAII(MetaProcessor* p);
      ~MaybeRedirectOutputRAII() { pop(); }
    private:
      void pop();
    };

  public:
    MetaProcessor(Interpreter& interp, llvm::raw_ostream& outs);
    ~MetaProcessor();

    const Interpreter& getInterpreter() const { return m_Interp; }

    llvm::raw_ostream& getOuts() const { return m_Outs; }

    ///\brief Process the input coming from the prompt and possibli returns
    /// result of the execution of the last statement
    /// @param[in] input_line - the user input
    /// @param[out] result - the cling::StoredValueRef as result of the
    ///             execution of the last statement
    /// @param[out] compRes - whether compilation was successful
    ///
    ///\returns 0 on success or the indentation of the next input line should
    /// have in case of multi input mode.
    ///\returns -1 if quit was requiested.
    ///
    int process(const char* input_line,
                Interpreter::CompilationResult& compRes,
                cling::StoredValueRef* result);

    ///\brief When continuation is requested, this cancels and ignores previous
    /// input, resetting the continuation to a new line.
    void cancelContinuation() const;

    ///\brief Returns the number of imbalanced tokens seen in the current input.
    ///
    int getExpectedIndent() const;

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
                     Interpreter::CompilationResult& compRes,
                     cling::StoredValueRef* result);

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
                      StoredValueRef* result,
                      bool ignoreOutmostBlock = false);

    ///\brief Set the stdout and stderr stream to the appropriate file.
    ///
    ///\param [in] file - The file for teh redirection
    ///\param [in] stream - Which stream to redirect: stdout, stderr or both.
    ///\param [in] append - Write in append mode.
    ///
    void setStdStream(llvm::StringRef file, RedirectionScope stream,
                      bool append);
  };
} // end namespace cling

#endif // CLING_METAPROCESSOR_H
