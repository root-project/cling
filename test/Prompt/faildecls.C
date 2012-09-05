// RUN: cat %s | %cling -Xclang -verify

struct {int j;}; // expected-error {{anonymous structs and classes must be class members}}

.q
