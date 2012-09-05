//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: Interpreter.cpp 45775 2012-08-31 14:54:11Z vvassilev $
// author:  Vassil Vassilev <vvasilev@cern.ch>
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
