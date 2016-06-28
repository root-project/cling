#ifndef CLINGTABCOMPLETION_H
#define CLINGTABCOMPLETION_H

#include <string>
#include <vector>

namespace cling {

  class Interpreter;

  class ClingTabCompletion {
    const cling::Interpreter& ParentInterp;
  
  public:
  	ClingTabCompletion(cling::Interpreter& Parent) : ParentInterp(Parent) {}
	~ClingTabCompletion() {}

  	bool Complete(const std::string& Line /*in+out*/,
                  size_t& Cursor /*in+out*/,
                  std::vector<std::string>& DisplayCompletions /*out*/);
  };
}
#endif
