// RUN: cat %s | %cling | FileCheck %s

//This file checks a pointer load operation for null prt dereference.
int *p;
int x;
x = *p; // CHECK: Warning: you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]
n
