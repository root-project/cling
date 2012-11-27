//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "InputValidator.h"

#include "PunctuationLexer.h"

//#include "clang/Lex/Preprocessor.h"

namespace cling {
  InputValidator::ValidationResult
  InputValidator::validate(llvm::StringRef line) {
    ValidationResult Res = kComplete;

    // FIXME: Do it properly for the comments too
    if (!line.startswith("//") 
        && !line.startswith("/*") && !line.startswith("*/")) {
      PunctuationLexer PL(line);
      Token Tok;
      do {
        PL.LexPunctuator(Tok);
        int kind = (int)Tok.getKind();

        // If there is " or ' we don't need to look for balancing until we 
        // enounter matching " or '
        if (kind >= (int)tok::quote && kind <= (int)tok::apostrophe) {
          if (m_ParenStack.empty())
            m_ParenStack.push(kind);
          else if (m_ParenStack.top() == kind)
            m_ParenStack.pop();
          else
            continue;
        }

        // In case when we need closing brace.
        if (kind >= (int)tok::l_square && kind <= (int)tok::r_brace) {
          // The closing paren kind is open paren kind + 1 (i.e odd number)
          if (kind % 2) {
            // closing the right one?
            if (m_ParenStack.empty()) {
              Res = kMismatch;
              break;
            }
            int prev = m_ParenStack.top();
            if (prev != kind - 1) {
              Res = kMismatch;
              break;
            }
            m_ParenStack.pop();
          } 
          else
            m_ParenStack.push(kind);
        }
      }
      while (Tok.isNot(tok::eof));
    }

    if (!m_ParenStack.empty() && Res != kMismatch)
      Res = kIncomplete;

    if (!m_Input.empty()) {
      if (!m_ParenStack.empty() && (m_ParenStack.top() == tok::quote 
                                    || m_ParenStack.top() == tok::apostrophe))
        m_Input.append("\\n");
      else 
        m_Input.append("\n");
    }
    else
      m_Input = "";

    m_Input.append(line);

    return Res;
  }

  void InputValidator::reset() {
    m_Input = "";
    while (!m_ParenStack.empty())
      m_ParenStack.pop();
  }
} // end namespace cling
