// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// XFAIL: *
// The main issue is that expected - error is not propagated to the source file and
// the expected diagnostics get misplaced.

.storeState "testMetaProcessor"

.x CannotDotX.h() // expected-error@2 {{use of undeclared identifier 'CannotDotX'}}
// CHECK: Error in cling::MetaProcessor: execute file failed.
.x CannotDotX.h()
// CHECK: Error in cling::MetaProcessor: execute file failed.
// expected-error@3 3 {{redefinition of 'MyClass'}}
// expected-error@4 3 {{expected member name or ';' after declaration specifiers}}
// expected-note@3 3 {{previous definition is here}}

// Here we cannot revert MyClass from CannotDotX.h
.L CannotDotX.h
// CHECK: Error in cling::MetaProcessor: load file failed.
.L CannotDotX.h
// CHECK: Error in cling::MetaProcessor: load file failed.

.compareState "testMetaProcessor"
// CHECK-NOT: File with AST differencies stored in: testMetaProcessorAST.diff
.q
