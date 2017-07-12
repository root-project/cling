//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test blockComments

(/*
   1
   2
   3 */
  8
  // single line 1
  /* single line 2*/
)
// CHECK: (int) 8

// Check nested indentation works
/*
 {
  (
   [
   ]
  )
 }
*/

// Check nested indentation doesn't error on mismatched closures
/*
  {
    [
      (
      }
    )
  ]
*/

( 5
  /*
   + 1
   + 2
   + 3 */
  + 4
  // single line 1
  /* single line 2*/
)
// CHECK-NEXT: (int) 9

/*
  This should work
  // As should this // */

/*
  This will warn
  // /* */ // expected-warning {{within block comment}}

.rawInput 1
*/ // expected-error {{expected unqualified-id}}
.rawInput 0


// This is a side effect of wrapping, expression is compiled as */; so 2 errors
*/ // expected-error@2 {{expected expression}} expected-error@3 {{expected expression}}

/* // /* */// expected-warning {{within block comment}}

/* // /* *// */
// expected-warning@input_line_27:2 {{within block comment}}
// expected-error@input_line_27:2 {{expected expression}}
// expected-error@input_line_27:2 {{expected expression}}
// expected-error@input_line_27:3 {{expected expression}}

/* //  *  // */

// Check preprocessor blocked out
/*
#if 1

#else er
#we not gonna terminate this
  #include "stop messing around.h"
#finished

*/

// Check meta-commands are blocked out
/*
  .L SomeStuff
  .x some more
  .q
*/

( 5
  /*
   + 10
   + 20 */
  /*
    + 30
  */
  + 4
  // single line 1
  + 10
  /* single line 2*/
  /* ) */
)
// CHECK-NEXT: (int) 19

/* 8 + */ 9 /* = 20 */
// CHECK-NEXT: (int) 9

/*
// Check inline asteriks
*******************************************************
*    Row   * Instance *   fEvents.fEventNo * fShowers *
*******************************************************
*        0 *        0 *                  0 *       10 *
*        0 *        1 *                  0 *       20 *
*        0 *        2 *                  2 *       30 *
*******************************************************
*/

// Check inline slashes
/*
A/B
*/

32
// CHECK-NEXT: (int) 32

/* Check inline asteriks ****/
62
// CHECK-NEXT: (int) 62

// ROOT-8529
12/3*4
// CHECK-NEXT: (int) 16

// ROOT-7354
/*
    * :(
*/
42
// CHECK-NEXT: (int) 42

(1/1)*1
// CHECK-NEXT: (int) 1


int A = 5, B = 25, *Ap = &A;
B / *Ap
// CHECK-NEXT: (int) 5

.q
