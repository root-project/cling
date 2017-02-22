//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

// Checks:
// Savannah #99210 https://savannah.cern.ch/bugs/index.php?99210
// Savannah #99234 https://savannah.cern.ch/bugs/?99234

// Let's start with simpler example pointing out the issue:
int i = 1; i++; int j = i;
j
// CHECK: (int) 2

extern "C" int printf(const char*,...);
.rawInput
class RAII {
public:
   RAII(int i) { I = new int(i); printf("RAII%d\n", ++InstanceCount); };
   int incr() { return ++(*I); }
   int get() { return *I; }
   ~RAII() { delete I; printf("~RAII%d\n", InstanceCount--); }
private:
   RAII(RAII&);
   RAII& operator=(RAII);
   int* I;
   static int InstanceCount; // will notice object copy
};
int RAII::InstanceCount = 0;
.rawInput

// This works because each line ends up in a separate wrapper
RAII R(12); // CHECK: RAII1
R.get();
int res = R.incr() // CHECK: 13

// This does not work because the decls and their inits are run before the
// call to R2.incr(), i.e. the second statement in the line.
// Savannah #99210 https://savannah.cern.ch/bugs/index.php?99210
RAII R2(42);R2.incr();int res2 = R2.get()
// CHECK: RAII2
// CHECK: 43
.q

// CHECK: ~RAII2
// CHECK: ~RAII1
// Enforce that only two objects got ever constructed:
// CHECK-NOT: ~RAII0
