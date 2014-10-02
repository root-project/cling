//RUN: make -C %testexecdir/../../clang/ test TESTSUITE=Sema CLANG=%p/clangTestUnloader.sh
//XFAIL: *

// Current SemaCXX failures:
// Expected Passes    : 392
// Expected Failures  : 1
// Unexpected Failures: 184

// Current Sema failures:
// Expected Passes    : 331
// Unexpected Failures: 90
