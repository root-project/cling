// RUN: cat %s | %cling -Xclang -verify -I%p
// XFAIL: vg
.x CannotDotX.h() // expected-error {{use of undeclared identifier 'CannotDotX'}} 
.x CannotDotX.h() // expected-error {{use of undeclared identifier 'CannotDotX'}}

 // Uses a bug in the implementation of .L, which must be fixed soon.
 // Exposes another issue with the VerifyDiagnosticConsumer in the context of 
 // cling. The first .L shouldn't cause errors . However when using the
 // preprocessor most probably we lose track of the errors source locations
 // and files.
.L CannotDotX.h "// expected-error {{redefinition of 'MyClass'}} expected-error {{expected member name or ';' after declaration specifiers}} expected-node {{previous definition is here}}


.L CannotDotX.h

.q
