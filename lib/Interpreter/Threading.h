//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Philippe Canal <pcanal@fnal.gov>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_THREADING_H
#define CLING_THREADING_H


#include <atomic>

namespace cling {
namespace internal {

  class SpinLockGuard {
    // Trivial spin lock guard

  private:
    ///\brief The atomic flag to be used as a spin lock.
    ///
    std::atomic_flag& m_Flag;

  public:
    ///\brief Spin until we until the flag becomes (or is) false
    ///
    SpinLockGuard(std::atomic_flag& aflag) : m_Flag(aflag)
    {
      while (m_Flag.test_and_set(std::memory_order_acquire));
    }

    //\brief Release the lock by setting the false to false.
    ~SpinLockGuard()
    {
      m_Flag.clear(std::memory_order_release);
    }

  };

} // end of internal namespace
} // end of cling namespace

#endif // end of CLING_THREADING_H
