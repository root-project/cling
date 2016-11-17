#ifndef Def2sub_h
#define Def2sub_h

#include "nesting/h1.h"

class CtorWithDefault {
public:
  CtorWithDefault(int i = 0) {};
};

// Supporting the default paramter here, requires an
// extension of the information stored with the annotation
// so that the cleanup is triggered upon any inclusion of
// of Def2sub.h rather than only the one coming from Def2.h
template <typename T = int>
class TemplateWithAllDefault
{
};

#endif
