//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/MetaProcessor.h"

#include "Display.h"
#include "InputValidator.h"
#include "MetaParser.h"
#include "MetaSema.h"
#include "cling/Interpreter/Interpreter.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Path.h"

#include <fstream>
#include <cstdlib>
#include <cctype>


using namespace clang;

namespace cling {

  MetaProcessor::MetaProcessor(Interpreter& interp) : m_Interp(interp) {
    m_InputValidator.reset(new InputValidator());
    m_MetaParser.reset(new MetaParser(new MetaSema(interp, *this)));
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
    m_MetaParser->enterNewInputLine(input_line);
    if (m_MetaParser->isMetaCommand()) {

      if (m_MetaParser->isQuitRequested())
        return -1;

      //TODO: set the compilation result if there was error in the meta commands
      return expectedIndent;
    }

    // Check if the current statement is now complete. If not, return to
    // prompt for more.
    if (m_InputValidator->validate(input_line) == InputValidator::kIncomplete) {
      if (compRes)
        *compRes = Interpreter::kMoreInputExpected;
      return m_InputValidator->getExpectedIndent();
    }

    //  We have a complete statement, compile and execute it.
    std::string input = m_InputValidator->getInput();
    m_InputValidator->reset();
    Interpreter::CompilationResult compResLocal;
    // if (m_Options.RawInput)
    //   compResLocal = m_Interp.declare(input);
    // else
    compResLocal = m_Interp.process(input, result);
    if (compRes) *compRes = compResLocal;

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
                                  StoredValueRef* result,
                               Interpreter::CompilationResult* compRes /*=0*/) {
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
      if (result)
        interpRes = m_Interp.evaluate(expression, *result);
      else
        interpRes = m_Interp.execute(expression);
      m_CurrentlyExecutingFile = llvm::StringRef();
      if (topmost)
        m_TopExecutingFile = llvm::StringRef();
    }
    if (compRes) *compRes = interpRes;
    return (interpRes != Interpreter::kFailure);
  }

  Interpreter::CompilationResult
  MetaProcessor::readInputFromFile(llvm::StringRef filename,
                                 StoredValueRef* result /* = 0 */,
                                 bool ignoreOutmostBlock /*=false*/) {

    {
      // check that it's not binary:
      std::ifstream in(filename.str().c_str(), std::ios::in | std::ios::binary);
      char magic[1024] = {0};
      in.read(magic, sizeof(magic));
      size_t readMagic = in.gcount();
      if (readMagic >= 4) {
        llvm::sys::LLVMFileType fileType
          = llvm::sys::IdentifyFileType(magic, 4);
        if (fileType != llvm::sys::Unknown_FileType) {
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
              llvm::errs() << "Error in cling::MetaProcessor: missing closing '}'!\n";
              // be confident, just go on.
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
    Interpreter::CompilationResult ret = Interpreter::kSuccess;
    process(content.c_str(), result, &ret);
    m_CurrentlyExecutingFile = llvm::StringRef();
    if (topmost)
      m_TopExecutingFile = llvm::StringRef();
    return ret;
  }

} // end namespace cling
