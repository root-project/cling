//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "InputValidator.h"

#include "MetaLexer.h"

namespace cling {
  InputValidator::ValidationResult
  InputValidator::validate(llvm::StringRef line) {
    ValidationResult Res = kComplete;

    Token Tok;
    const char* curPos = line.data();
    do {
      // If the input is "some//string" or '//some/string' we need to disable
      // the comment checks.
      bool hasQuoteOrApostrophe = false;
      for (unsigned i = 0, e = m_ParenStack.size(); i < e; ++i) {
        int entry = m_ParenStack[i];
        if (entry == (int)tok::quote || entry == (int)tok::apostrophe) {
          hasQuoteOrApostrophe = true;
          break;
        }
      }
      MetaLexer::LexPunctuatorAndAdvance(curPos, Tok,
                                         /*skipComments*/hasQuoteOrApostrophe);
      int kind = (int)Tok.getKind();

      // if (kind == tok::hash) {
      //   // Handle #ifdef ...
      //   //          ...
      //   //        #endif
      //   MetaLexer Lexer(curPos);
      //   Lexer.Lex(Tok);
      //   if (Tok.getKind() == tok::ident) {
      //     if (Tok.getIdent().startswith("if")) {
      //       Res = kIncomplete;
      //       m_ParenStack.push_back(kind);
      //     }
      //     else if (Tok.getIdent().startswith("end")) {
      //       assert(m_ParenStack.back() == kind && "No coresponding # to pop?");
      //       m_ParenStack.pop_back();
      //     }
      //   }
      // }

      if (kind >= (int)tok::quote && kind <= (int)tok::apostrophe) {
        // If there is " or ' we don't need to look for balancing until we
        // enounter matching " or '
        Res = kIncomplete;
        if (m_ParenStack.size() && m_ParenStack.back() == kind) {
          // We know that the top of the stack will have the leading quote or
          // apostophe because nobody is pushing onto it until we find a balance
          m_ParenStack.pop_back();
          Res = kComplete;
        }
        if (!hasQuoteOrApostrophe)
          m_ParenStack.push_back(kind); // Register the first " or '
        continue;
      }
      // All the rest is irrelevant in QuoteOrApostrophe mode.
      if (hasQuoteOrApostrophe)
        continue;

      // In case when we need closing brace.
      if (kind >= (int)tok::l_square && kind <= (int)tok::r_brace) {
        // The closing paren kind is open paren kind + 1 (i.e odd number)
        if (kind % 2) {
          // closing the right one?
          if (m_ParenStack.empty()) {
            Res = kMismatch;
            break;
          }
          int prev = m_ParenStack.back();
          if (prev != kind - 1) {
            Res = kMismatch;
            break;
          }
          m_ParenStack.pop_back();
        }
        else
          m_ParenStack.push_back(kind);
      }
    }
    while (Tok.isNot(tok::eof));

    if (!m_ParenStack.empty() && Res != kMismatch)
      Res = kIncomplete;

    if (!m_Input.empty()) {
      if (!m_ParenStack.empty() && (m_ParenStack.back() == tok::quote
                                    || m_ParenStack.back() == tok::apostrophe))
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
    m_ParenStack.clear();
  }
} // end namespace cling
