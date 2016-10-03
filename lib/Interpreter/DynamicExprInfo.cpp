//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicExprInfo.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace cling {
namespace runtime {
  namespace internal {
    const char* DynamicExprInfo::getExpr() {
      int i = 0;
      size_t found;

      llvm::SmallString<256> Buf;
      while ((found = m_Result.find("@")) && (found != std::string::npos)) {
        Buf.resize(0);
        llvm::raw_svector_ostream Strm(Buf);
        Strm << m_Addresses[i];
        m_Result = m_Result.insert(found + 1, Strm.str());
        m_Result = m_Result.erase(found, 1);
        ++i;
      }

      return m_Result.c_str();
    }
  } // end namespace internal
} // end namespace runtime
} // end namespace cling
