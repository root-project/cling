//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | env CLING_HOME="%S/Inputs" %cling %s 2>&1 | FileCheck %s
// UNSUPPORTED: system-windows

// CHECK: Startup file ran, magic # was 43210

void StartupFile() {
  io::cout << "The magic # is " << startup_magic_num << '\n';
  // CHECK: The magic # is 43213
  io::cout << "The magic string is " << startup_magic_str << '\n';
  // CHECK: The magic string is AaBbCc__51
}

// expected-no-diagnostics
