// RUN: cat %s | %cling -I%p | FileCheck %s

#include "SymbolResolverCallback.h"

.dynamicExtensions

gCling->setCallbacks(new cling::test::SymbolResolverCallback(gCling));

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
