//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INPUT_VALIDATOR_H
#define CLING_INPUT_VALIDATOR_H

#include "clang/Basic/TokenKinds.h"

#include "llvm/ADT/StringRef.h"

#include <stack>

namespace clang {
  class LangOptions;
}

namespace cling {

  ///\brief Provides storage for the input and tracks down whether the (, [, {
  /// are balanced.
  ///
  class InputValidator {
  private:
    ///\brief The input being collected.
    ///
    std::string m_Input;

    ///\brief Stack used for checking the brace balance.
    ///
    std::deque<int> m_ParenStack;

  public:
    InputValidator() {}
    ~InputValidator() {}

    ///\brief Brace balance validation could encounter.
    ///
    enum ValidationResult {
      kIncomplete, ///< There is dangling brace.
      kComplete, ///< All braces are in balance.
      kMismatch ///< Closing brace doesn't match to opening. Eg: void f(};
    };

    ///\brief Checks whether the input contains balanced number of braces
    ///
    ///\param[in] line - Input line to validate.
    ///\returns Information about the outcome of the validation.
    ///
    ValidationResult validate(llvm::StringRef line);

    ///\returns Reference to the collected input.
    ///
    std::string& getInput() {
      return m_Input;
    }

    ///\brief Retrieves the number of spaces that the next input line should be
    /// indented.
    ///
    int getExpectedIndent() { return m_ParenStack.size(); }

    ///\brief Resets the collected input and its corresponding brace stack.
    ///
    void reset();
  };
}
#endif // CLING_INPUT_VALIDATOR_H
