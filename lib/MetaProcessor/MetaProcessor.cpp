//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/MetaProcessor.h"

#include "Display.h"
#include "InputValidator.h"
#include "MetaParser.h"
#include "MetaSema.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Path.h"

#include <fstream>
#include <cstdlib>
#include <cctype>
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#define STDIN_FILENO  0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif

using namespace clang;

namespace cling {

  MetaProcessor::MaybeRedirectOutputRAII::MaybeRedirectOutputRAII(
                                          MetaProcessor* p)
  :m_MetaProcessor(p), m_isCurrentlyRedirecting(0) {
    StringRef redirectionFile;
    m_MetaProcessor->increaseRedirectionRAIILevel();
    if (!m_MetaProcessor->m_PrevStdoutFileName.empty()) {
      redirectionFile = m_MetaProcessor->m_PrevStdoutFileName.back();
      redirect(stdout, redirectionFile.str(), kSTDOUT);
    }
    if (!m_MetaProcessor->m_PrevStderrFileName.empty()) {
      redirectionFile = m_MetaProcessor->m_PrevStderrFileName.back();
      // Deal with the case 2>&1 and 2&>1
      if (strcmp(redirectionFile.data(), "_IO_2_1_stdout_") == 0) {
        // If out is redirected to a file.
        if (!m_MetaProcessor->m_PrevStdoutFileName.empty()) {
          redirectionFile = m_MetaProcessor->m_PrevStdoutFileName.back();
        } else {
          unredirect(m_MetaProcessor->m_backupFDStderr, STDERR_FILENO, stderr);
        }
      }
      redirect(stderr, redirectionFile.str(), kSTDERR);
    }
  }

  MetaProcessor::MaybeRedirectOutputRAII::~MaybeRedirectOutputRAII() {
    pop();
    m_MetaProcessor->decreaseRedirectionRAIILevel();
  }

  void MetaProcessor::MaybeRedirectOutputRAII::redirect(FILE* file,
                                        const std::string& fileName,
                                        MetaProcessor::RedirectionScope scope) {
    if (!fileName.empty()) {
      FILE* redirectionFile = freopen(fileName.c_str(), "a", file);
      if (!redirectionFile) {
        llvm::errs()<<"cling::MetaProcessor::MaybeRedirectOutputRAII::redirect:"
                    " Not succefully reopened the redirection file "
                    << fileName.c_str() << "\n.";
      } else {
        m_isCurrentlyRedirecting |= scope;
      }
    }
  }

  void MetaProcessor::MaybeRedirectOutputRAII::pop() {
    //If we have only one redirection RAII
    //only then do the unredirection.
    if (m_MetaProcessor->getRedirectionRAIILevel() != 1)
      return;

    if (m_isCurrentlyRedirecting & kSTDOUT) {
      unredirect(m_MetaProcessor->m_backupFDStdout, STDOUT_FILENO, stdout);
    }
    if (m_isCurrentlyRedirecting & kSTDERR) {
      unredirect(m_MetaProcessor->m_backupFDStderr, STDERR_FILENO, stderr);
    }
  }

  void MetaProcessor::MaybeRedirectOutputRAII::unredirect(int backupFD,
                                                          int expectedFD,
                                                          FILE* file) {
    // Switch back to previous file after line is processed.

    // Flush the current content if there is any.
    if (!feof(file)) {
      fflush(file);
    }
    // Copy the original fd for the std.
    if (dup2(backupFD, expectedFD) != expectedFD) {
        llvm::errs() << "cling::MetaProcessor::unredirect "
                     << "The unredirection file descriptor not valid "
                     << backupFD << ".\n";
    }
  }

  MetaProcessor::MetaProcessor(Interpreter& interp, raw_ostream& outs)
    : m_Interp(interp), m_Outs(&outs) {
    m_InputValidator.reset(new InputValidator());
    m_MetaParser.reset(new MetaParser(new MetaSema(interp, *this)));
    m_backupFDStdout = copyFileDescriptor(STDOUT_FILENO);
    m_backupFDStderr = copyFileDescriptor(STDERR_FILENO);
  }

  MetaProcessor::~MetaProcessor() {}

  int MetaProcessor::process(const char* input_text,
                             Interpreter::CompilationResult& compRes,
                             Value* result) {
    if (result)
      *result = Value();
    compRes = Interpreter::kSuccess;
    int expectedIndent = m_InputValidator->getExpectedIndent();

    if (expectedIndent)
      compRes = Interpreter::kMoreInputExpected;
    if (!input_text || !input_text[0]) {
      // nullptr / empty string, nothing to do.
      return expectedIndent;
    }
    std::string input_line(input_text);
    if (input_line == "\n") { // just a blank line, nothing to do.
      return expectedIndent;
    }
    //  Check for and handle meta commands.
    m_MetaParser->enterNewInputLine(input_line);
    MetaSema::ActionResult actionResult = MetaSema::AR_Success;
    if (m_MetaParser->isMetaCommand(actionResult, result)) {

      if (m_MetaParser->isQuitRequested())
        return -1;

      if (actionResult != MetaSema::AR_Success)
        compRes = Interpreter::kFailure;
       // ExpectedIndent might have changed after meta command.
       return m_InputValidator->getExpectedIndent();
    }

    // Check if the current statement is now complete. If not, return to
    // prompt for more.
    if (m_InputValidator->validate(input_line) == InputValidator::kIncomplete) {
      compRes = Interpreter::kMoreInputExpected;
      return m_InputValidator->getExpectedIndent();
    }

    //  We have a complete statement, compile and execute it.
    std::string input = m_InputValidator->getInput();
    m_InputValidator->reset();
    // if (m_Options.RawInput)
    //   compResLocal = m_Interp.declare(input);
    // else
    compRes = m_Interp.process(input, result);

    return 0;
  }

  void MetaProcessor::cancelContinuation() const {
    m_InputValidator->reset();
  }

  int MetaProcessor::getExpectedIndent() const {
    return m_InputValidator->getExpectedIndent();
  }

  Interpreter::CompilationResult
  MetaProcessor::readInputFromFile(llvm::StringRef filename,
                                   Value* result,
                                   bool ignoreOutmostBlock /*=false*/) {

    {
      // check that it's not binary:
      std::ifstream in(filename.str().c_str(), std::ios::in | std::ios::binary);
      char magic[1024] = {0};
      in.read(magic, sizeof(magic));
      size_t readMagic = in.gcount();
      if (readMagic >= 4) {
        llvm::StringRef magicStr(magic,in.gcount());
        llvm::sys::fs::file_magic fileType
          = llvm::sys::fs::identify_magic(magicStr);
        if (fileType != llvm::sys::fs::file_magic::unknown) {
          llvm::errs() << "Error in cling::MetaProcessor: "
            "cannot read input from a binary file!\n";
          return Interpreter::kFailure;
        }
        unsigned printable = 0;
        for (size_t i = 0; i < readMagic; ++i)
          if (isprint(magic[i]))
            ++printable;
        if (10 * printable <  5 * readMagic) {
          // 50% printable for ASCII files should be a safe guess.
          llvm::errs() << "Error in cling::MetaProcessor: "
            "cannot read input from a (likely) binary file!\n" << printable;
          return Interpreter::kFailure;
        }
      }
    }

    std::ifstream in(filename.str().c_str());
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    std::string content(size, ' ');
    in.seekg(0);
    in.read(&content[0], size);

    if (ignoreOutmostBlock && !content.empty()) {
      static const char whitespace[] = " \t\r\n";
      std::string::size_type posNonWS = content.find_first_not_of(whitespace);
      // Handle comments before leading {
      while (posNonWS != std::string::npos
             && content[posNonWS] == '/' && content[posNonWS+1] == '/') {
        // Remove the comment line
        posNonWS = content.find_first_of('\n', posNonWS+2)+1;
      }
      std::string::size_type replaced = posNonWS;
      if (posNonWS != std::string::npos && content[posNonWS] == '{') {
        // hide the curly brace:
        content[posNonWS] = ' ';
        // and the matching closing '}'
        posNonWS = content.find_last_not_of(whitespace);
        if (posNonWS != std::string::npos) {
          if (content[posNonWS] == ';' && content[posNonWS-1] == '}') {
            content[posNonWS--] = ' '; // replace ';' and enter next if
          }
          if (content[posNonWS] == '}') {
            content[posNonWS] = ' '; // replace '}'
          } else {
            std::string::size_type posBlockClose = content.find_last_of('}');
            if (posBlockClose != std::string::npos) {
              content[posBlockClose] = ' '; // replace '}'
            }
            std::string::size_type posComment
              = content.find_first_not_of(whitespace, posBlockClose);
            if (posComment != std::string::npos
                && content[posComment] == '/' && content[posComment+1] == '/') {
              // More text (comments) are okay after the last '}', but
              // we can not easily find it to remove it (so we need to upgrade
              // this code to better handle the case with comments or
              // preprocessor code before and after the leading { and
              // trailing })
              while (posComment <= posNonWS) {
                content[posComment++] = ' '; // replace '}' and comment
              }
            } else {
              content[replaced] = '{';
              // By putting the '{' back, we keep the code as consistent as
              // the user wrote it ... but we should still warn that we not
              // goint to treat this file an unamed macro.
              llvm::errs()
                << "Warning in cling::MetaProcessor: can not find the closing '}', "
                << llvm::sys::path::filename(filename)
                << " is not handled as an unamed script!\n";
            } // did not find "//"
          } // remove comments after the trailing '}'
        } // find '}'
      } // have '{'
    } // ignore outmost block

    std::string strFilename(filename.str());
    m_CurrentlyExecutingFile = strFilename;
    bool topmost = !m_TopExecutingFile.data();
    if (topmost)
      m_TopExecutingFile = m_CurrentlyExecutingFile;
    Interpreter::CompilationResult ret;
    // We don't want to value print the results of a unnamed macro.
    content = "#line 2 \"" + filename.str() + "\" \n" + content;
    if (process((content + ";").c_str(), ret, result)) {
      // Input file has to be complete.
       llvm::errs()
          << "Error in cling::MetaProcessor: file "
          << llvm::sys::path::filename(filename)
          << " is incomplete (missing parenthesis or similar)!\n";
      ret = Interpreter::kFailure;
    }
    m_CurrentlyExecutingFile = llvm::StringRef();
    if (topmost)
      m_TopExecutingFile = llvm::StringRef();
    return ret;
  }

  void MetaProcessor::setFileStream(llvm::StringRef file, bool append, int fd,
              llvm::SmallVector<llvm::SmallString<128>, 2>& prevFileStack) {
    // If we have a fileName to redirect to store it.
    if (!file.empty()) {
      prevFileStack.push_back(file);
      // pop and push a null terminating 0.
      // SmallVectorImpl<T> does not have a c_str(), thus instead of casting to
      // a SmallString<T> we null terminate the data that we have and pop the
      // 0 char back.
      prevFileStack.back().push_back(0);
      prevFileStack.back().pop_back();
      if (!append) {
        FILE * f;
        if (!(f = fopen(file.data(), "w"))) {
          llvm::errs() << "cling::MetaProcessor::setFileStream:"
                       " The file path " << file.data() << "is not valid.";
        } else {
          fclose(f);
        }
      }
    // Else unredirection, so switch to the previous file.
    } else {
      // If there is no previous file on the stack we pop the file
      if (!prevFileStack.empty()) {
        prevFileStack.pop_back();
      }
    }
  }

  void MetaProcessor::setStdStream(llvm::StringRef file,
                                   RedirectionScope stream, bool append) {

    if (stream & kSTDOUT) {
      setFileStream(file, append, STDOUT_FILENO, m_PrevStdoutFileName);
    }
    if (stream & kSTDERR) {
      setFileStream(file, append, STDERR_FILENO, m_PrevStderrFileName);
    }
  }

  int MetaProcessor::copyFileDescriptor(int fd) {
    int backupFD = dup(fd);
    if (backupFD < 0) {
      llvm::errs() << "MetaProcessor::copyFileDescriptor: Duplicating the file"
                   " descriptor " << fd << "resulted in an error."
                   " Will not be able to unredirect.";
    }
    return backupFD;
  }

  void MetaProcessor::registerUnloadPoint(const Transaction* T,
                                          llvm::StringRef filename) {
    m_MetaParser->getActions().registerUnloadPoint(T, filename);
  }

} // end namespace cling
