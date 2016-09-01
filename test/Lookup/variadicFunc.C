//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -fno-rtti 2>&1 | FileCheck %s
// XFAIL:*

// This test assures that varidiac funcions can be found by our string-based
// lookup.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"

#include <cstdio>
#include <string>

using namespace std;
using namespace llvm;

//
//  We need to fetch the global scope declaration,
//  otherwise known as the translation unit decl.
//
const cling::LookupHelper& lookup = gCling->getLookupHelper();
const clang::Decl* G = lookup.findScope("", cling::LookupHelper::WithDiagnostics);
printf("G: 0x%lx\n", (unsigned long) G);
//CHECK: G: 0x{{[1-9a-f][0-9a-f]*$}}

std::string buf;
clang::PrintingPolicy Policy(G->getASTContext().getPrintingPolicy());

#include <iostream>
#include <cstdarg>
void simple_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            std::cout << i << '\n';
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            std::cout << static_cast<char>(c) << '\n';
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            std::cout << d << '\n';
        }
        ++fmt;
    }

    va_end(args);
}

const clang::FunctionDecl* variadicF = lookup.findFunctionArgs(G, "simple_printf", "const char*, ...", cling::LookupHelper::WithDiagnostics);
printf("simple_printf: 0x%lx\n", (unsigned long) variadicF);
//CHECK-NEXT: simple_printf: 0x{{[1-9a-f][0-9a-f]*$}}

.q
