//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "InputValidator.h"

#include "clang/Lex/Preprocessor.h"

using namespace clang;

namespace cling {
  InputValidator::ValidationResult
  InputValidator::validate(llvm::StringRef line, LangOptions& LO) {
    if (!m_Input.empty())
      m_Input.append("\n");
    else
      m_Input = "";

    m_Input.append(line);

    llvm::MemoryBuffer* MB = llvm::MemoryBuffer::getMemBuffer(line);
    Lexer RawLexer(SourceLocation(), LO, MB->getBufferStart(),
                   MB->getBufferStart(), MB->getBufferEnd());
    Token Tok;
    do {
      RawLexer.LexFromRawLexer(Tok);
      int kind = (int)Tok.getKind();
      if (kind >= (int)tok::l_square
          && kind <= (int)tok::r_brace) {
        kind -= (int)tok::l_square;
        if (kind % 2) {
          // closing the right one?
          if (m_ParenStack.empty()) return kMismatch;
          int prev = m_ParenStack.top();
          if (prev != kind - 1) return kMismatch;
          m_ParenStack.pop();
        } else {
          m_ParenStack.push(kind);
        }
      }
    }
    while (Tok.isNot(tok::eof));
    if (!m_ParenStack.empty())
      return kIncomplete;

    return kComplete;
  }

  void InputValidator::reset() {
    m_Input = "";
    while (!m_ParenStack.empty())
      m_ParenStack.pop();
  }
} // end namespace cling
