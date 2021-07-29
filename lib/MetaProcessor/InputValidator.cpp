//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/InputValidator.h"

#include "cling/MetaProcessor/MetaLexer.h"

#include <algorithm>

namespace cling {
  bool InputValidator::inBlockComment() const {
    return !m_ParenStack.empty() && m_ParenStack.back() == tok::l_comment;
  }

  InputValidator::ValidationResult
  InputValidator::validate(llvm::StringRef line) {
    ValidationResult Res = kComplete;
    MetaLexer Lex(line.data(), /*skipWhiteSpace=*/true);
    Token Tok, lastNonSpaceTok;

    // Only check for 'template' if we're not already indented
    if (m_ParenStack.empty()) {
      MetaLexer::RAII RAII(Lex);
      Lex.Lex(Tok);
      if (Tok.is(tok::ident) && Tok.getIdent() == "template")
        m_ParenStack.push_back(tok::less);
    }

    do {
      if (Tok.isNot(tok::space))
        lastNonSpaceTok = Tok;
      Lex.Lex(Tok);

      // In multiline comments, ignore anything that is not the `*/` token
      if (inBlockComment() && Tok.isNot(tok::r_comment))
        continue;

      switch (Tok.getKind()) {
      default:
        break;
      case tok::comment:
        Lex.ReadToEndOfLine(Tok);
        break;
      case tok::r_comment:
        if (inBlockComment())
          m_ParenStack.pop_back();
        else
          Res = kMismatch;
        break;

      case tok::l_square: case tok::l_paren: case tok::l_brace:
      case tok::l_comment:
        m_ParenStack.push_back(Tok.getKind());
        break;
      case tok::r_square: case tok::r_paren: case tok::r_brace:
        {
          auto tos = m_ParenStack.empty()
            ? tok::unknown : static_cast<tok::TokenKind>(m_ParenStack.back());
          if (!Tok.closesBrace(tos)) {
            Res = kMismatch;
            break;
          }
          m_ParenStack.pop_back();
          // '}' will also pop a template '<' if their is one
          if (Tok.getKind() == tok::r_brace && m_ParenStack.size() == 1
              && m_ParenStack.back() == tok::less)
            m_ParenStack.pop_back();
        }
        break;

      case tok::semicolon:
        // Template forward declatation, i.e. 'template' '<' ... '>' ... ';'
        if (m_ParenStack.size() == 1 && m_ParenStack.back() == tok::less)
          m_ParenStack.pop_back();
        break;

      case tok::hash:
        Lex.SkipWhitespace();
        Lex.LexAnyString(Tok);
        const llvm::StringRef PPtk = Tok.getIdent();
        if (PPtk.startswith("if")) {
          m_ParenStack.push_back(tok::hash);
        } else if (PPtk.startswith("endif") &&
                   (PPtk.size() == 5 || PPtk[5]=='/' || isspace(PPtk[5]))) {
            if (m_ParenStack.empty() || m_ParenStack.back() != tok::hash)
              Res = kMismatch;
            else
              m_ParenStack.pop_back();
        }
        break;
      }
    } while (Tok.isNot(tok::eof) && Res != kMismatch);

    const bool Continue = (lastNonSpaceTok.getKind() == tok::backslash
                           || lastNonSpaceTok.getKind() == tok::comma);
    if (Continue || (!m_ParenStack.empty() && Res != kMismatch))
      Res = kIncomplete;

    if (!m_Input.empty()) {
      m_Input.append("\n");
    }
    m_Input.append(line);
    return Res;
  }

  void InputValidator::reset(std::string* input) {
    if (input) {
      assert(input->empty() && "InputValidator::reset got non empty argument");
      input->swap(m_Input);
    } else
      std::string().swap(m_Input);

    std::deque<int>().swap(m_ParenStack);
  }
} // end namespace cling
