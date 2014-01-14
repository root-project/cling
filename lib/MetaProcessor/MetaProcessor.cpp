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
#include "cling/Interpreter/StoredValueRef.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Path.h"

#include <fstream>
#include <cstdlib>
#include <cctype>
#include <unistd.h>
#include <stdio.h>


using namespace clang;

namespace cling {

  MetaProcessor::MaybeRedirectOutputRAII::MaybeRedirectOutputRAII(MetaProcessor* p)
  :m_MetaProcessor(p), m_PrevStdoutFileName(""), m_PrevStderrFileName("") {
    //Empty file acts as a flag.
    redirect(STDOUT_FILENO, m_PrevStdoutFileName, m_MetaProcessor->m_FileOut, stdout);
    // Deal with the case 2>&1 and 2&>1
    if (strcmp(m_MetaProcessor->m_FileErr.c_str(), "_IO_2_1_stdout_") == 0) {
      SmallString<1024> stdoutName;
      cacheStd(STDERR_FILENO, stdoutName);
      m_MetaProcessor->m_FileErr = stdoutName.c_str();
    }
    //Empty file acts as a flag.
    redirect(STDERR_FILENO, m_PrevStderrFileName, m_MetaProcessor->m_FileErr, stderr);

  }

  void MetaProcessor::MaybeRedirectOutputRAII::pop() {

    unredirect(m_PrevStdoutFileName, stdout);
    unredirect(m_PrevStderrFileName, stderr);
  }

  bool MetaProcessor::MaybeRedirectOutputRAII::cacheStd(int fd,
                                                      llvm::SmallVectorImpl<char>& file) {

    int ttyname_Result = ttyname_r(fd, const_cast<char*>(file.data()), file.capacity());
    while (ttyname_Result == ERANGE) {
      file.reserve(16*file.capacity());
      ttyname_Result = ttyname_r(fd, const_cast<char*>(file.data()), file.capacity());
    }

    if (ttyname_Result == 0) {
      file.set_size(strlen(file.data()));
      return true;
    } else if (ttyname_Result == EBADF) {
      llvm::errs() << "Error in cling::MetaProcessor: Bad file descriptor.";
    } else if (ttyname_Result == ENOTTY) {
      llvm::errs() << "File descriptor does not refer to a terminal device.";
    } else if (ttyname_Result == EAGAIN) {
      llvm::errs() << "The device driver was in use by another process, or the driver"
                   << "was unable to carry out the request due to an outstanding command in progress.";
    } else if (ttyname_Result == EINTR) {
      llvm::errs() << "The function was interrupted by a signal.";
    } else if (ttyname_Result == ENOSYS) {
      llvm::errs() << "The ttyname_r() function isn't implemented for the filesystem specified by filedes.";
    } else if (ttyname_Result == EPERM) {
      llvm::errs() << "The process doesn't have sufficient permission to carry out the requested command.";
    }
    return false;
  }

  void MetaProcessor::MaybeRedirectOutputRAII::redirect(int fd,
                                                        llvm::SmallVectorImpl<char>& prevFile,
                                                        std::string fileName,
                                                        struct _IO_FILE * standard) {
    if (!fileName.empty()) {
      //Cache prevous stdout.
      if (!cacheStd(fd, prevFile)) {
        llvm::errs() << "cling::MetaProcessor Error: Was not able to cache "
                     << "previous output. Will not be able to redirect back.";
      }
      _IO_FILE * redirectionFile = freopen(fileName.c_str(), "a", standard);
      if (!redirectionFile) {
        llvm::errs() << "cling::MetaProcessor Error: The file path is not valid.";
      } else {
        standard = redirectionFile;
      }
    }
  }

  void MetaProcessor::MaybeRedirectOutputRAII::unredirect(llvm::SmallVectorImpl<char>& file,
                                                          struct _IO_FILE * standard) {
    //Switch back to standard output after line is processed.
    if (!file.empty()) {
      _IO_FILE * redirectionFile = freopen(file.data(), "w", standard);
      if (!redirectionFile) {
        llvm::errs() << "cling::MetaProcessor Error: The file path is not valid.";
      } else {
        standard = redirectionFile;
      }
    }
  }


  MetaProcessor::MetaProcessor(Interpreter& interp, raw_ostream& outs) 
    : m_Interp(interp), m_Outs(outs), m_FileOut(""), m_FileErr("") {
    m_InputValidator.reset(new InputValidator());
    m_MetaParser.reset(new MetaParser(new MetaSema(interp, *this)));
  }

  MetaProcessor::~MetaProcessor() {}

  int MetaProcessor::process(const char* input_text,
                             Interpreter::CompilationResult& compRes,
                             StoredValueRef* result) {
    if (result)
      *result = StoredValueRef::invalidValue();
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
      return expectedIndent;
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

  // Run a file: .x file[(args)]
  bool MetaProcessor::executeFile(llvm::StringRef file, llvm::StringRef args,
                                  Interpreter::CompilationResult& compRes,
                                  StoredValueRef* result) {
    // Look for start of parameters:
    typedef std::pair<llvm::StringRef,llvm::StringRef> StringRefPair;

    StringRefPair pairPathFile = file.rsplit('/');
    if (pairPathFile.second.empty()) {
       pairPathFile.second = pairPathFile.first;
    }
    StringRefPair pairFuncExt = pairPathFile.second.rsplit('.');

    Interpreter::CompilationResult interpRes = m_Interp.loadFile(file);
    if (interpRes == Interpreter::kSuccess) {
      std::string expression = pairFuncExt.first.str() + "(" + args.str() + ")";
      m_CurrentlyExecutingFile = file;
      bool topmost = !m_TopExecutingFile.data();
      if (topmost)
        m_TopExecutingFile = m_CurrentlyExecutingFile;
      
      interpRes = m_Interp.process(expression, result);

      m_CurrentlyExecutingFile = llvm::StringRef();
      if (topmost)
        m_TopExecutingFile = llvm::StringRef();
    }
    compRes = interpRes;
    return (interpRes != Interpreter::kFailure);
  }

  Interpreter::CompilationResult
  MetaProcessor::readInputFromFile(llvm::StringRef filename,
                                 StoredValueRef* result,
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
      std::string::size_type replaced = posNonWS;
      if (posNonWS != std::string::npos) {
        if (content[posNonWS] == '{') {
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
              // More text (comments) are okay after the last '}', but
              // we can not easily find it to remove it (so we need to upgrade
              // this code to better handle the case with comments or
              // preprocessor code before and after the leading { and
              // trailing })
              content[replaced] = '{';
              // By putting the '{' back, we keep the code as consistent as
              // the user wrote it ... but we should still warn that we not 
              // goint to treat this file an unamed macro.
              llvm::errs() 
               << "Warning in cling::MetaProcessor: can not find the closing '}', "
               << llvm::sys::path::filename(filename)
               << " is not handled as an unamed script!\n";
            }
          } // find '}'
        } // have '{'
      } // have non-whitespace
    } // ignore outmost block

    std::string strFilename(filename.str());
    m_CurrentlyExecutingFile = strFilename;
    bool topmost = !m_TopExecutingFile.data();
    if (topmost)
      m_TopExecutingFile = m_CurrentlyExecutingFile;
    Interpreter::CompilationResult ret;
    if (process(content.c_str(), ret, result)) {
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

  void MetaProcessor::setStdStream(llvm::StringRef file,
                                   RedirectionScope stream, bool append) {

    if (stream & kSTDOUT) {
      m_FileOut = file;
      if (!append && !m_FileOut.empty()) {
        FILE* f = fopen(m_FileOut.c_str(), "w");
      }
    }
    if (stream & kSTDERR) {
      m_FileErr = file;
      if (!append && !m_FileErr.empty()) {
        FILE* f = fopen(m_FileErr.c_str(), "w");
      }
    }
  }

} // end namespace cling
