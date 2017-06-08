//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling -C -E -P  %s | %cling -nostdinc++ -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cling -C -E -P -DCLING_VALEXTRACT_ERR %s | %cling -nostdinc++ -nobuiltininc -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cling -C -E -P -DCLING_VALEXTRACT_ERR2 %s | %cling -nostdinc++ -nobuiltininc -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cling -C -E -P -DCLING_VALEXTRACT_ERR3 %s | %cling -nostdinc++ -nobuiltininc -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cling -C -E -P -DCLING_VALEXTRACT_ERR4 %s | %cling -nostdinc++ -nobuiltininc -Xclang -verify 2>&1 | FileCheck %s

// expected-error@input_line_1:1 {{'new' file not found}}

//      CHECK: Warning in cling::IncrementalParser::CheckABICompatibility():
// CHECK-NEXT:  Failed to extract C++ standard library version.


#if defined(CLING_VALEXTRACT_ERR) || defined(CLING_VALEXTRACT_ERR2) || \
    defined(CLING_VALEXTRACT_ERR3) || defined(CLING_VALEXTRACT_ERR4)

struct Trigger {} Tr;

#ifdef CLING_VALEXTRACT_ERR
Tr // expected-error@2 {{ValueExtractionSynthesizer could not find: 'cling namespace'.}}
#endif

#ifdef CLING_VALEXTRACT_ERR2
namespace cling {}
Tr // expected-error@2 {{ValueExtractionSynthesizer could not find: 'cling::runtime namespace'.}}
#endif

#ifdef CLING_VALEXTRACT_ERR3
namespace cling { namespace runtime {} }
Tr // expected-error@2 {{ValueExtractionSynthesizer could not find: 'cling::runtime::gCling'.}}
#endif

#ifdef CLING_VALEXTRACT_ERR4
namespace cling { namespace runtime { void* gCling; namespace internal { void* setValueWithAlloc; void* setValueNoAlloc; } } }
Tr // expected-error@2 {{ValueExtractionSynthesizer could not find: 'cling::runtime::internal::copyArray'.}}
// Make sure not to crash on subsequent attempts
Tr // expected-error@2 {{ValueExtractionSynthesizer could not find: 'cling::runtime::internal::copyArray'.}}
#endif

#endif

.q
