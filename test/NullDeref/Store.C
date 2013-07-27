// RUN: cat %s | %cling -Xclang -verify

//This file checks a pointer store operation for null ptr dereference.
int *p;
*p = 6; 
n
