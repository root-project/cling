//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// clang-format off
// RUN: %rm /tmp/__testing_cling_history
// RUN: cat %s | env --unset=CLING_NOHISTORY CLING_HISTSIZE=8 CLING_HISTFILE="/tmp/__testing_cling_history" %cling - 2>&1
// RUN: diff /tmp/__testing_cling_history "%S/Inputs/cling_history"
// UNSUPPORTED: system-windows

#include <iostream>

int i = 8;

template<class T>
struct type_a {
	const T t;
	double d;

/*
		comment
 * */

	static const int i = 37;
};

// another comment

using std::cout;

std::cout << "test\n";

