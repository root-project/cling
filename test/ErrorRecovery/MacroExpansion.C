// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#define BEGIN_NAMESPACE namespace test_namespace {
#define END_NAMESPACE }

.storeState "testMacroExpansion"
 .rawInput 1

BEGIN_NAMESPACE int j; END_NAMESPACE
BEGIN_NAMESPACE int j; END_NAMESPACE // expected-error {{redefinition of 'j'}} expected-note {{previous definition is here}} 

.rawInput 0
.compareState "testMacroExpansion"
// CHECK-NOT: File with AST differencies stored in: testMacroExpansionAST.diff
// Make FileCheck happy with having at least one positive rule: 
int a = 5
// CHECK: (int) 5
.q
