//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify
// XFAIL: vanilla-cling
// Test FwdPrinterTest

// Test similar to ROOT-7037
// Equivalent to parsing the dictionary preamble
.T Def2b.h fwd_Def2b.h
#include "fwd_Def2b.h"
// Then doing the autoparsing
#include "Def2b.h"
A<int> ai2;
// And then do both a second time with a different template instantiation.
.T Def2c.h fwd_Def2c.h
#include "fwd_Def2c.h"
#include "Def2c.h"

// In some implementations the AutoloadingVisitor was stripping the default
// template parameter value from the class template definition leading to
// compilation error at this next line:
A<float> af2;


// We want to make sure that forward template carying default are not
// affected and that forward declaration are properly cleaned-up (if needed).
template <typename T = int> class DefaultInFwd;
template <typename T> class WithDefaultAndFwd;

.T Def2.h fwd_Def2.h
#include "fwd_Def2.h"

// In some implementation the AutoloadingVisitor, when called upon by the next
// #includes, was not properly removing default value that was attached to
// this following class template forward declaration (the default comes from
// the forward declaration in fwd_Def2.h). See ROOT-8443.
template <typename T> class TemplateWithUserForward;

// The includsion of nesting/h2.h is not (yet) supported as it is included
// in the forward declaration generation step via a #include with a path
// relative to its includer (h1.h) and this we can not properly record
// that this is the same thing as this one (or at least it is harder than
// what we do for now).
// #include "nesting/h2.h"
#include "nesting/h1.h"
#include "Def2sub.h"
#include "Def2.h"

DefaultInFwd<> dif;
WithDefaultAndFwd<> wdaf;
TemplateWithUserForward<> twuf;
TemplateWithAllDefault<> twad;
WithDefaultInH1<> wdh1;
WithDefaultInH2<> wdh2;

// In some implementation the AutoloadingVisitor was not when Def2sub.h, which
// contains the definition for CtorWithDefault, and then the implementation
// was also looping over all element of the decl chain without skipping definition,
// resulting in a loss of the default parameter values for the method/functions of
// CtorWithDefault when AutoloadingVisitor was called upon because of the inclusion
// Def2.h
CtorWithDefault c;
M::N::A mna;
M::N::D mnd;

.T Enum.h fwd_enums.h
#include "fwd_enums.h"
#include "Enum.h"

//expected-no-diagnostics
.q
