#ifndef TMPL_COLLIDING_H
#define TMPL_COLLIDING_H
#include "Tmpl.h"
// Specializations Tmpl<b_NN::X> that stay lazy in this module. They share the
// lazy-specialization lookup hash with Tmpl<a::X>, which only takes the
// unqualified name of the argument's record into account.
#define SPEC(n) namespace b##n { struct X; using T = Tmpl<X>; }
#define SPEC10(n) SPEC(n##0) SPEC(n##1) SPEC(n##2) SPEC(n##3) SPEC(n##4) \
                  SPEC(n##5) SPEC(n##6) SPEC(n##7) SPEC(n##8) SPEC(n##9)
#define SPEC100(n) SPEC10(n##0) SPEC10(n##1) SPEC10(n##2) SPEC10(n##3) \
                   SPEC10(n##4) SPEC10(n##5) SPEC10(n##6) SPEC10(n##7) \
                   SPEC10(n##8) SPEC10(n##9)
SPEC100(_)
#endif
