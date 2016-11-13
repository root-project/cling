// In the ROOT case, the duplicated default parameter definition is suppressed
// when it is due to two annotated forward decl (i.e. the $clingAutoload$ ones)
// Since in standalone cling this is not the case, let's 'emulate' that behavior
// by remove the default from the declaration.

//#include "Def2a.h"
#ifndef DEF2_A
template<class T, class U>
class A{};
#endif

A<float,int> ac;
