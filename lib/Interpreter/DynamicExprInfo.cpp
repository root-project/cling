//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicExprInfo.h"

#include <sstream>

namespace cling {
namespace runtime {
  namespace internal {
    const char* DynamicExprInfo::getExpr() {
      int i = 0;
      size_t found;

      while ((found = m_Result.find("@")) && (found != std::string::npos)) {
        std::stringstream address;
        address << m_Addresses[i];
        m_Result = m_Result.insert(found + 1, address.str());
        m_Result = m_Result.erase(found, 1);
        ++i;
      }

      return m_Result.c_str();
    }
  } // end namespace internal
} // end namespace runtime
} // end namespace cling
