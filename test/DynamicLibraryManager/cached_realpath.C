//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// REQUIRES: not_system-windows

// RUN: %mkdir %t-dir
// RUN: cd %t-dir

// RUN: %mkdir %t-dir/dir1/dir11/dir111
// RUN: %mkdir %t-dir/dir1/dir11/dir112
// RUN: %mkdir %t-dir/dir1/dir12/dir121
// RUN: %mkdir %t-dir/dir1/dir12/dir122
// RUN: %mkdir %t-dir/dir2/dir21/dir211
// RUN: %mkdir %t-dir/dir2/dir21/dir212

// RUN: echo "a" > %t-dir/dir1/a.txt
// RUN: echo "b" > %t-dir/dir1/dir11/b.txt
// RUN: echo "c" > %t-dir/dir1/dir11/dir111/c.txt
// RUN: echo "d" > %t-dir/dir1/dir12/d.txt
// RUN: echo "e" > %t-dir/dir2/dir21/dir211/e.txt

// RUN: ln -f -s %t-dir/dir1/dir11 dir1/linkdir11
// RUN: ln -f -s -r %t-dir/dir1/dir11 dir1/rlinkdir11
// RUN: ln -f -s %t-dir/dir1/dir12 dir1/linkdir12
// RUN: ln -f -s %t-dir/dir1/dir11/dir111 dir1/dir11/linkdir111
// RUN: ln -f -s -r %t-dir/dir1/dir11/dir111 dir1/dir11/rlinkdir111

// RUN: ln -f -s %t-dir/dir1/dir11/dir111/c.txt dir1/dir11/dir111/linkc.txt
// RUN: ln -f -s -r %t-dir/dir1/dir11/dir111/c.txt dir1/dir11/dir111/rlinkc.txt

// RUN: ln -f -s %t-dir/dir1/dir11/dir111/nofile.txt dir1/dir11/dir111/linknofile.txt
// RUN: ln -f -s -r %t-dir/dir1/dir11/dir111/nofile.txt dir1/dir11/dir111/rlinknofile.txt

// RUN: ln -f -s %t-dir/dir2/dir21 dir2/linkdir21
// RUN: ln -f -s %t-dir/dir2/linkdir21 dir2/linkdir21a
// RUN: ln -f -s -r %t-dir/dir2/dir21 dir2/rlinkdir21
// RUN: ln -f -s -r %t-dir/dir2/rlinkdir21 dir2/rlinkdir21a
// RUN: ln -f -s %t-dir/dir2/linkdir21 dir2/linkdir21a1

// RUN: ln -f -s %t-dir/dir2 dir2/dir21/linkdir2
// RUN: ln -f -s -r %t-dir/dir2 dir2/dir21/rlinkdir2

// RUN: ln -f -s -r %t-dir/dir2/dir21/dir211 dir2/rlinkdir211
// RUN: ln -f -s %t-dir/dir2/dir21/dir211 dir2/linkdir211
// RUN: ln -f -s -r %t-dir/dir2/dir21/dir211/e.txt dir2/rlinke.txt
// RUN: ln -f -s %t-dir/dir2/dir21/dir211/e.txt dir2/linke.txt

// RUN: ln -f -s %t-dir/dir2/selfinfloop.txt dir2/selfinfloop.txt
// RUN: ln -f -s %t-dir/dir2/infloop2.txt dir2/infloop1.txt
// RUN: ln -f -s %t-dir/dir2/infloop1.txt dir2/infloop2.txt
// RUN: ln -f -s -r %t-dir/dir2/rselfinfloop.txt dir2/rselfinfloop.txt

// RUN: ln -f -s -r %t-dir/dir1/a.txt dir2/backtoa.txt


// RUN: cat %s | %cling -fno-rtti 2>&1 | FileCheck %s


#include <string>
#include <iostream>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#include "../lib/Interpreter/DynamicLibraryManagerSymbol.cpp"

.rawInput 1
void test_realpath(std::string path) {
  // system realpath
  errno = 0;
  char system_resolved_path[4096];
  system_resolved_path[0] = '\0';
  realpath(path.c_str(), system_resolved_path);
  int err_s = errno;
  if (err_s !=0 ) system_resolved_path[0] = '\0';

  // cached_realpath
  errno = 0;
  std::string cached_resolved_path = cached_realpath(path);
  int err_c = errno;

  if (err_s != err_c || std::string(system_resolved_path) != cached_resolved_path) {
    std::cout << "realpath: " << path.c_str() << "\n";
    std::cout << "  err_s=" << err_s << ", rp_s=" << system_resolved_path << "\n";
    std::cout << "  err_c=" << err_c << ", rp_c=" << cached_resolved_path.c_str() << "\n\n";
  } else if (err_c != 0) {
    std::cout << "ERROR\n";
  } else {
    std::cout << "OK\n";
  }
  std::cout << std::flush;
}
.rawInput 0

// Test: cached_realpath

test_realpath(""); // CHECK: ERROR

test_realpath("/"); // CHECK: OK
test_realpath("/tmp"); // CHECK: OK

test_realpath("."); // CHECK: OK
test_realpath(".."); // CHECK: OK
test_realpath("../"); // CHECK: OK
test_realpath("../."); // CHECK: OK
test_realpath("/."); // CHECK: OK
test_realpath("/.."); // CHECK: OK
test_realpath("/../"); // CHECK: OK
test_realpath("/../.."); // CHECK: OK

//test_realpath("~"); // OK
//test_realpath("~/tmp"); // OK

test_realpath("dir1"); // CHECK: OK
test_realpath("dir1/a.txt"); // CHECK: OK
test_realpath("dir1/dir11/b.txt"); // CHECK: OK

test_realpath("nodir"); // CHECK: ERROR
test_realpath("dir1/nodir/b.txt"); // CHECK: ERROR

test_realpath("dir1/linkdir11/b.txt"); // CHECK: OK
test_realpath("dir1/rlinkdir11/b.txt"); // CHECK: OK
test_realpath("dir1/linkdir11/dir111/c.txt"); // CHECK: OK
test_realpath("dir1/rlinkdir11/dir111/c.txt"); // CHECK: OK

test_realpath("dir1/linkdir11/dir111/nofile.txt"); // CHECK: ERROR
test_realpath("dir1/rlinkdir11/dir111/nofile.txt"); // CHECK: ERROR
test_realpath("dir1/linkdir12/nofile.txt"); // CHECK: ERROR

test_realpath("dir1/dir11/dir111/linkc.txt"); // CHECK: OK
test_realpath("dir1/dir11/dir111/rlinkc.txt"); // CHECK: OK
test_realpath("dir1/linkdir11/dir111/linkc.txt"); // CHECK: OK
test_realpath("dir1/linkdir11/dir111/rlinkc.txt"); // CHECK: OK
test_realpath("dir1/rlinkdir11/dir111/linkc.txt"); // CHECK: OK
test_realpath("dir1/rlinkdir11/dir111/rlinkc.txt"); // CHECK: OK

test_realpath("dir1/dir11/dir111/linknofile.txt"); // CHECK: ERROR
test_realpath("dir1/dir11/dir111/rlinknofile.txt"); // CHECK: ERROR
test_realpath("dir1/linkdir11/dir111/linknofile.txt"); // CHECK: ERROR
test_realpath("dir1/linkdir11/dir111/rlinknofile.txt"); // CHECK: ERROR

test_realpath("dir2/linkdir211/."); // CHECK: OK
test_realpath("dir2/rlinkdir211/."); // CHECK: OK
test_realpath("dir2/linkdir211/.."); // CHECK: OK
test_realpath("dir2/rlinkdir211/.."); // CHECK: OK

test_realpath("dir2/dir21/linkdir2"); // CHECK: OK
test_realpath("dir2/dir21/rlinkdir2"); // CHECK: OK
test_realpath("dir2/linkdir21a1"); // CHECK: OK
test_realpath("dir2/rlinkdir21a"); // CHECK: OK

test_realpath("dir2/dir21/dir211/dir211"); // CHECK: ERROR
test_realpath("dir2/dir21/dir211/dir211/e.txt"); // CHECK: ERROR
test_realpath("dir2/dir21/dir211/dir211/dir211"); // CHECK: ERROR
test_realpath("dir2/dir21/dir211/dir211/dir211/e.txt"); // CHECK: ERROR
test_realpath("dir2/dir21/dir211/dir211/dir211/dir211"); // CHECK: ERROR
test_realpath("dir2/dir21/dir211/dir211/dir211/dir211/e.txt"); // CHECK: ERROR
test_realpath("dir2/linke.txt"); // CHECK: OK
test_realpath("dir2/rlinke.txt"); // CHECK: OK

test_realpath("dir2/./backtoa.txt"); // CHECK: OK
test_realpath("dir2/dir21/../backtoa.txt"); // CHECK: OK
test_realpath("dir2//backtoa.txt"); // CHECK: OK

test_realpath("../nofile.txt"); // CHECK: ERROR

test_realpath("dir2/infloop1.txt"); // CHECK: ERROR
test_realpath("dir2/infloop2.txt"); // CHECK: ERROR
test_realpath("dir2/selfinfloop.txt"); // CHECK: ERROR
test_realpath("dir2/rselfinfloop.txt"); // CHECK: ERROR

.q
