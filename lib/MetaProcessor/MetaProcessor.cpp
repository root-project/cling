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
#include <sstream>
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

  class MetaProcessor::RedirectOutput {
    typedef std::stack<std::string> RedirectStack;
    enum { kNumRedirects = 2 };

    int m_Backups[kNumRedirects];
    RedirectStack m_Stacks[kNumRedirects];

    static int translate(int fd) {
      switch (fd) {
        case STDOUT_FILENO: return 0;
        case STDERR_FILENO: return 1;
        default: break;
      }
      llvm_unreachable("Cannot backup given file descriptor");
      return -1;
    }

    void backup(int fd, int iFD) {
      // Have we already backed it up?
      if (m_Backups[iFD] == 0) {
        m_Backups[iFD] = dup(fd);
        if (m_Backups[iFD] < 0) {
          llvm::errs() << "RedirectOutput::copyFileDescriptor: Duplicating the"
                          " file descriptor " << fd << " resulted in an error."
                          " Will not be able to unredirect.";
        }
      }
    }

  public:
    RedirectOutput() {
      ::memset(m_Backups, 0, sizeof(m_Backups));
    }

    ~RedirectOutput() {
      for (unsigned i = 0; i < kNumRedirects; ++i) {
        if (const int fd = m_Backups[i])
          ::close(fd);
      }
    }

    void redirect(llvm::StringRef filePath, bool append, int fd) {
      const int iFD = translate(fd);
      RedirectStack& rStack = m_Stacks[iFD];

      // If we have a fileName to redirect to store it.
      if (!filePath.empty()) {
        backup(fd, iFD);
        rStack.push(filePath.str());
        if (!append) {
          FILE * f;
          if (!(f = fopen(rStack.top().c_str(), "w"))) {
            llvm::errs() << "cling::MetaProcessor::setFileStream:"
                            " The file path " << filePath << "is not valid.";
          } else
            fclose(f);
        }
      } else {
        // Else unredirection, so switch to the previous file.
        // If there is no previous file on the stack we pop the file
        if (!rStack.empty())
          rStack.pop();
      }
    }

    void unredirect(int expectedFD, FILE* file) {
      // Switch back to previous file after line is processed.
      // Flush the current content if there is any.
      if (!feof(file))
        fflush(file);

      const int iFD = translate(expectedFD);
      // Copy the original fd for the std.
      if (dup2(m_Backups[iFD], expectedFD) != expectedFD) {
          llvm::errs() << "cling::MetaProcessor::unredirect "
                       << "The unredirection file descriptor not valid "
                       << m_Backups[iFD] << ".\n";
      }
    }
    
    const std::string* file(int fd) const {
      const int iFD = translate(fd);
      return m_Stacks[iFD].empty() ? nullptr : &m_Stacks[iFD].top();
    }
  };

  MetaProcessor::MaybeRedirectOutputRAII::MaybeRedirectOutputRAII(
                                          MetaProcessor* p)
  :m_MetaProcessor(p), m_isCurrentlyRedirecting(0) {
    StringRef redirectionFile;
    if (RedirectOutput* output = m_MetaProcessor->m_RedirectOutput.get()) {
      const std::string* stdOut = output->file(STDOUT_FILENO);
      if (stdOut)
        redirect(stdout, *stdOut, kSTDOUT);
      if (const std::string* stdErr = output->file(STDERR_FILENO)) {
        if (*stdErr == "_IO_2_1_stdout_") {
          // If out is redirected to a file.
          if (!stdOut)
            output->unredirect(STDERR_FILENO, stderr);
          else
            stdErr = stdOut;
        }
        redirect(stderr, *stdErr, kSTDERR);
      }
    }
  }

  MetaProcessor::MaybeRedirectOutputRAII::~MaybeRedirectOutputRAII() {
    pop();
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
    if (RedirectOutput* redirect = m_MetaProcessor->m_RedirectOutput.get()) {
      if (m_isCurrentlyRedirecting & kSTDOUT)
        redirect->unredirect(STDOUT_FILENO, stdout);
      if (m_isCurrentlyRedirecting & kSTDERR)
        redirect->unredirect(STDERR_FILENO, stderr);
    }
  }

  MetaProcessor::MetaProcessor(Interpreter& interp, raw_ostream& outs)
    : m_Interp(interp), m_Outs(&outs) {
    m_InputValidator.reset(new InputValidator());
    m_MetaParser.reset(new MetaParser(new MetaSema(interp, *this)));
  }

  MetaProcessor::~MetaProcessor() {
  }

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
    if (!m_InputValidator->inBlockComment() &&
         m_MetaParser->isMetaCommand(actionResult, result)) {

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
    std::string input;
    m_InputValidator->reset(&input);
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

  static Interpreter::CompilationResult reportIOErr(llvm::StringRef File,
                                                    const char* What) {
    llvm::errs() << "Error in cling::MetaProcessor: "
          "cannot " << What << " input: '" << File << "'\n";
    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult
  MetaProcessor::readInputFromFile(llvm::StringRef filename,
                                   Value* result,
                                   size_t posOpenCurly,
                                   bool lineByLine) {

    // FIXME: This will fail for Unicode BOMs (and seems really weird)
    {
      // check that it's not binary:
      std::ifstream in(filename.str().c_str(), std::ios::in | std::ios::binary);
      if (in.fail())
        return reportIOErr(filename, "open");

      char magic[1024] = {0};
      in.read(magic, sizeof(magic));
      size_t readMagic = in.gcount();
      // Binary files < 300 bytes are rare, and below newlines etc make the
      // heuristic unreliable.
      if (!in.fail() && readMagic >= 300) {
        llvm::StringRef magicStr(magic,in.gcount());
        llvm::sys::fs::file_magic fileType
          = llvm::sys::fs::identify_magic(magicStr);
        if (fileType != llvm::sys::fs::file_magic::unknown)
          return reportIOErr(filename, "read from binary");

        unsigned printable = 0;
        for (size_t i = 0; i < readMagic; ++i)
          if (isprint(magic[i]))
            ++printable;
        if (10 * printable <  5 * readMagic) {
          // 50% printable for ASCII files should be a safe guess.
          return reportIOErr(filename, "won't read from likely binary");
        }
      }
    }

    std::ifstream in(filename.str().c_str());
    if (in.fail())
      return reportIOErr(filename, "open");

    in.seekg(0, std::ios::end);
    if (in.fail())
      return reportIOErr(filename, "seek");

    size_t size = in.tellg();
    if (in.fail())
      return reportIOErr(filename, "tell");

    in.seekg(0);
    if (in.fail())
      return reportIOErr(filename, "rewind");

    std::string content(size, ' ');
    in.read(&content[0], size);
    if (in.fail())
      return reportIOErr(filename, "read");

    if (posOpenCurly != (size_t)-1 && !content.empty()) {
      assert(content[posOpenCurly] == '{'
             && "No curly at claimed position of opening curly!");
      // hide the curly brace:
      content[posOpenCurly] = ' ';
      // and the matching closing '}'
      static const char whitespace[] = " \t\r\n";
      size_t posCloseCurly = content.find_last_not_of(whitespace);
      if (posCloseCurly != std::string::npos) {
        if (content[posCloseCurly] == ';' && content[posCloseCurly-1] == '}') {
          content[posCloseCurly--] = ' '; // replace ';' and enter next if
        }
        if (content[posCloseCurly] == '}') {
          content[posCloseCurly] = ' '; // replace '}'
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
            while (posComment <= posCloseCurly) {
              content[posComment++] = ' '; // replace '}' and comment
            }
          } else {
            content[posCloseCurly] = '{';
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
    } // ignore outermost block

#ifndef NDEBUG
    m_CurrentlyExecutingFile = filename;
    bool topmost = !m_TopExecutingFile.data();
    if (topmost)
      m_TopExecutingFile = m_CurrentlyExecutingFile;
#endif

    content.insert(0, "#line 2 \"" + filename.str() + "\" \n");
    // We don't want to value print the results of a unnamed macro.
    if (content.back() != ';')
      content.append(";");

    Interpreter::CompilationResult ret = Interpreter::kSuccess;
    if (lineByLine) {
      int rslt = 0;
      std::string line;
      std::stringstream ss(content);
      while (std::getline(ss, line, '\n')) {
        rslt = process(line.c_str(), ret, result);
        if (ret == Interpreter::kFailure)
          break;
      }
      if (rslt) {
        llvm::errs() << "Error in cling::MetaProcessor: file "
                     << llvm::sys::path::filename(filename)
                     << " is incomplete (missing parenthesis or similar)!\n";
      }
    } else
      ret = m_Interp.process(content, result);

#ifndef NDEBUG
    m_CurrentlyExecutingFile = llvm::StringRef();
    if (topmost)
      m_TopExecutingFile = llvm::StringRef();
#endif
    return ret;
  }

  void MetaProcessor::setStdStream(llvm::StringRef file, RedirectionScope scope,
                                   bool append) {
    assert((scope & kSTDOUT || scope & kSTDERR) && "Invalid RedirectionScope");
    if (!m_RedirectOutput)
      m_RedirectOutput.reset(new RedirectOutput);
    if (scope & kSTDOUT)
      m_RedirectOutput->redirect(file, append, STDOUT_FILENO);
    if (scope & kSTDERR)
      m_RedirectOutput->redirect(file, append, STDERR_FILENO);
  }

  void MetaProcessor::registerUnloadPoint(const Transaction* T,
                                          llvm::StringRef filename) {
    m_MetaParser->getActions().registerUnloadPoint(T, filename);
  }

} // end namespace cling
