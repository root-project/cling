// RUN: cat %s | %cling | FileCheck %s
// XFAIL: *

// Checks:
// Savannah #99210 https://savannah.cern.ch/bugs/index.php?99210
// Savannah #99234 https://savannah.cern.ch/bugs/?99234

extern "C" int printf(const char*,...);
.rawInput
class RAII {
public:
   RAII(int i) { I = new int(i); printf("RAII%d\n", ++InstanceCount); };
   int incr() { return ++(*I); }
   int get() { return *I; }
   ~RAII() { delete I; printf("~RAII%d\n", InstanceCount--); }
private:
   RAII(RAII&) {throw;};
   RAII& operator=(RAII) {throw;}
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

// This fails due to printf not printing into stdout at destruction time
// but into some other file stream that happens to end up in the terminal
// but cannot be piped.
// Likely bug https://savannah.cern.ch/bugs/?99234
// CHECK: ~RAII2
// CHECK: ~RAII1
// Enforce that only two objects got ever constructed:
// CHECK-NOT: ~RAII0
