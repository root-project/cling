// RUN: cat %s | %cling --metastr=META | FileCheck %s

// Test setting of meta escape
METAhelp // CHECK: Cling meta commands usage
