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

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"

#include <memory>
#include <stdio.h>

namespace cling {

  class Interpreter;
  class InputValidator;
  class MetaParser;
  class Value;

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
    std::unique_ptr<InputValidator> m_InputValidator;

    ///\brief The parser used to parse our tiny "meta" language
    ///
    std::unique_ptr<MetaParser> m_MetaParser;

    ///\brief Currently executing file as passed into executeFile
    ///
    llvm::StringRef m_CurrentlyExecutingFile;

    ///\brief Outermost currently executing file as passed into executeFile
    ///
    llvm::StringRef m_TopExecutingFile;

    ///\brief The output stream being used for various purposes.
    ///
    llvm::raw_ostream* m_Outs;

    ///\brief The file descriptor of the copy of stdout.
    ///
    int m_backupFDStdout;

    ///\brief The file descriptor of the copy of stdout.
    ///
    int m_backupFDStderr;

    ///\brief Stores the stack for the redirect file paths for out.
    llvm::SmallVector<llvm::SmallString<128>, 2> m_PrevStdoutFileName;
    ///\brief Stores the stack for the redirect file paths for err.
    llvm::SmallVector<llvm::SmallString<128>, 2> m_PrevStderrFileName;

    //Counter to handle more than one redirection RAAI's
    int m_RedirectionRAIILevel = 0;

  public:
    enum RedirectionScope {
      kSTDOUT = 1,
      kSTDERR = 2,
      kSTDBOTH = 3
    };

  public:
    ///\brief Class to be created for each processing input to be
    /// able to redirect std.
    class MaybeRedirectOutputRAII {
    private:
      MetaProcessor* m_MetaProcessor;
      int m_isCurrentlyRedirecting;

    public:
      MaybeRedirectOutputRAII(MetaProcessor* p);
      ~MaybeRedirectOutputRAII();

      // Don't mess with m_RedirectionRAIILevel
      MaybeRedirectOutputRAII(const MaybeRedirectOutputRAII&) = delete;
      MaybeRedirectOutputRAII(MaybeRedirectOutputRAII&&) = delete;
      MaybeRedirectOutputRAII& operator=(const MaybeRedirectOutputRAII&) = delete;
      MaybeRedirectOutputRAII& operator=(MaybeRedirectOutputRAII&&) = delete;

    private:
      void pop();
      void redirect(FILE* file, const std::string& fileName,
                    MetaProcessor::RedirectionScope scope);
      void unredirect(int backFD, int expectedFD, FILE* file);
    };

  public:
    MetaProcessor(Interpreter& interp, llvm::raw_ostream& outs);
    ~MetaProcessor();

    const Interpreter& getInterpreter() const { return m_Interp; }

    ///\brief Get the output stream used by the MetaProcessor for its output.
    /// (in contrast to the interpreter's output which is redirected using
    /// setStdStream()).
    llvm::raw_ostream& getOuts() const { return *m_Outs; }

    ///\brief Set the output stream used by the MetaProcessor for its output.
    /// (in contrast to the interpreter's output which is redirected using
    /// setStdStream()).
    ///
    ///\returns the address of the previous output stream, or 0 if it was unset.
    llvm::raw_ostream* setOuts(llvm::raw_ostream& outs) {
      llvm::raw_ostream* prev = m_Outs;
      m_Outs = &outs;
      return prev;
    }
    void increaseRedirectionRAIILevel() { m_RedirectionRAIILevel++; }

    void decreaseRedirectionRAIILevel() { m_RedirectionRAIILevel--; }

    int getRedirectionRAIILevel() { return m_RedirectionRAIILevel; }

    ///\brief Process the input coming from the prompt and possibli returns
    /// result of the execution of the last statement
    /// @param[in] input_line - the user input
    /// @param[out] result - the cling::Value as result of the
    ///             execution of the last statement
    /// @param[out] compRes - whether compilation was successful
    ///
    ///\returns 0 on success or the indentation of the next input line should
    /// have in case of multi input mode.
    ///\returns -1 if quit was requiested.
    ///
    int process(const char* input_line,
                Interpreter::CompilationResult& compRes,
                cling::Value* result);

    ///\brief When continuation is requested, this cancels and ignores previous
    /// input, resetting the continuation to a new line.
    void cancelContinuation() const;

    ///\brief Returns the number of imbalanced tokens seen in the current input.
    ///
    int getExpectedIndent() const;

    ///\brief Reads prompt input from file.
    ///
    ///\param [in] filename - The file to read.
    /// @param[out] result - the cling::Value as result of the
    ///             execution of the last statement
    ///\param [in] ignoreOutmostBlock - Whether to ignore enlosing {}.
    ///
    ///\returns result of the compilation.
    ///
    Interpreter::CompilationResult
    readInputFromFile(llvm::StringRef filename,
                      Value* result,
                      bool ignoreOutmostBlock = false);
    ///\brief Set the stdout and stderr stream to the appropriate file.
    ///
    ///\param [in] file - The file for the redirection.
    ///\param [in] stream - Which stream to redirect: stdout, stderr or both.
    ///\param [in] append - Write in append mode.
    ///
    void setStdStream(llvm::StringRef file, RedirectionScope stream,
                      bool append);

    ///\brief Register the file as an upload point for the Transaction T
    ///  when unloading that file, all transactions after T will be reverted.
    ///
    ///\param [in] T - the last transaction stay should filename be unloaded.
    ///\param [in] filename - The name of the file to be used as unload point.
    void registerUnloadPoint(const Transaction* T, llvm::StringRef filename);

  private:
    ///\brief Set a stream to a file
    ///
    ///\param [in] file - The file for the redirection.
    ///\param [in] append - Write in append mode.
    ///\param [in] fd - File descriptor for the file we want to open.
    ///\param [out] prevFileStack - The stack of previous file paths.
    void setFileStream(llvm::StringRef file, bool append, int fd,
                  llvm::SmallVector<llvm::SmallString<128> ,2>& prevFileStack);

    ///\brief Copy a file descriptor.
    ///
    ///\param [in] fd - The fd to be copied.

    ///\returns - The copy of the file descriptor.
    int copyFileDescriptor(int fd);
  };
} // end namespace cling

#endif // CLING_METAPROCESSOR_H
