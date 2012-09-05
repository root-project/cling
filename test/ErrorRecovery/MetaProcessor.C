// RUN: cat %s | %cling -Xclang -verify -I%p
// XFAIL: vg
// We expect for now this to fail, because the access to the invalid memory comes
// from the fact that we invalidate the included file cache and when we are in
// verify mode the verifier notifies about errors at the end of the source file.
//
// Tests the ChainedConsumer's ability to recover from errors. .x produces 
// #include \"CannotDotX.h\" \n void wrapper() {CannotDotX();}, which causes
// a TagDecl to be passed trough the consumers. This TagDecl is caught twice by
// the ChainedConsumer and cached is the queue of incoming declaration twice.
// If we encounter error the ChainedConsumer shouldn't try to remove the 
// declaration twice and this test makes sure of that.

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
