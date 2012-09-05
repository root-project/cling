// RUN: cat %s | %cling -Xclang -verify -I%p

#define BEGIN_NAMESPACE namespace test_namespace {
#define END_NAMESPACE }

.rawInput 1

BEGIN_NAMESPACE int j; END_NAMESPACE
BEGIN_NAMESPACE int j; END_NAMESPACE // expected-error {{redefinition of 'j'}} expected-note {{previous definition is here}} 

.rawInput 0
