//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %built_cling -I%p | FileCheck %s

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

.dynamicExtensions

std::unique_ptr<cling::test::SymbolResolverCallback> SRC;
SRC.reset(new cling::test::SymbolResolverCallback(gCling))
gCling->setCallbacks(std::move(SRC));

// Fixed size arrays
int a[5] = {1,2,3,4,5};
h->PrintArray(a, 5); // CHECK: 12345

float b[4][5] = {
  {1,2,3,4,5},
  {6,7,8,9,10},
  {11,12,13,14,15},
  {16,17,18,19,20},
};

h1->PrintArray(b, 4); // CHECK: 1234567891011121314151617181920

int c[3][4][5] = {
  {
    {1,2,3,4,5},
    {6,7,8,9,10},
    {11,12,13,14,15},
    {16,17,18,19,20},
  },

  {
    {1,2,3,4,5},
    {6,7,8,9,10},
    {11,12,13,14,15},
    {16,17,18,19,20},
  },

  {
    {1,2,3,4,5},
    {6,7,8,9,10},
    {11,12,13,14,15},
    {16,17,18,19,20},
  }
};

h2->PrintArray(c, 3); // CHECK: 123456789101112131415161718192012345678910111213141516171819201234567891011121314151617181920


.q
