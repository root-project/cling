#ifndef TABCOMPLETION_H
#define TABCOMPLETION_H

#include "textinput/Callbacks.h"

namespace textinput {
	class TabCompletion;
	class Text;
	class EditorRange;
}	

namespace cling {
  class TabCompletion : public textinput::TabCompletion {
    const cling::Interpreter& ParentInterp;
  
  public:
    TabCompletion(cling::Interpreter& Parent) : ParentInterp(Parent) {}

    ~TabCompletion() {}

    bool Complete(textinput::Text& Line /*in+out*/,
                size_t& Cursor /*in+out*/,
                textinput::EditorRange& R /*out*/,
                std::vector<std::string>& DisplayCompletions /*out*/) override {
      const InterpreterCallbacks* callbacks = ParentInterp.getCallbacks();
      callbacks->Complete(Line.GetText(), Cursor, DisplayCompletions); 
  }
  };
}
#endif
