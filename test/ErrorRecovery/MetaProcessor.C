// RUN: cat %s | %cling -Xclang -verify -I%p
// XFAIL: *
// The main issue is that expected-error is not propagated to the source file and
// the expected diagnostics get misplaced.
.x CannotDotX.h() // expected-error {{use of undeclared identifier 'CannotDotX'}} 
.x CannotDotX.h() // expected-error {{use of undeclared identifier 'CannotDotX'}}

 // Here we cannot revert MyClass from CannotDotX.h
.L CannotDotX.h // expected-error {{redefinition of 'MyClass'}} expected-error {{expected member name or ';' after declaration specifiers}} expected-node {{previous definition is here}}


.L CannotDotX.h

.q
