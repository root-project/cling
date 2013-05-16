# Module.mk for textinput module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME       := textinput
MODDIR        := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

TEXTINPUTDIR  := $(MODDIR)
TEXTINPUTDIRS := $(TEXTINPUTDIR)/src
TEXTINPUTDIRI := $(TEXTINPUTDIR)/inc

##### libTextInput - part of libCore #####
TEXTINPUTL    := $(MODDIRI)/LinkDef.h
TEXTINPUTDS   := $(call stripsrc,$(MODDIRS)/G__TextInput.cxx)
TEXTINPUTDO   := $(TEXTINPUTDS:.cxx=.o)
TEXTINPUTDH   := $(TEXTINPUTDS:.cxx=.h)

LTEXTINPUTS   := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/textinput/*.cpp))
LTEXTINPUTO   := $(call stripsrc,$(LTEXTINPUTS:.cpp=.o))

TEXTINPUTH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TEXTINPUTS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TEXTINPUTO    := $(call stripsrc,$(TEXTINPUTS:.cxx=.o)) $(LTEXTINPUTO)

TEXTINPUTDEP  := $(TEXTINPUTO:.o=.d) $(TEXTINPUTDO:.o=.d) $(LTEXTINPUTO:.o=.d)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TEXTINPUTH))

# include all dependency files
INCLUDEFILES  += $(TEXTINPUTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TEXTINPUTDIRI)/%.h
		cp $< $@

$(LTEXTINPUTO): %.o: $(ROOT_SRCDIR)/%.cpp
		$(MAKEDIR)
		$(MAKEDEP) -R -f$*.d -Y -w 1000 -- $(CXXFLAGS) -D__cplusplus -- $<
		$(CXX) $(OPT) $(CXXFLAGS) -I$(TEXTINPUTDIRS) $(CXXOUT)$@ -c $<

$(TEXTINPUTDS): $(TEXTINPUTH) $(TEXTINPUTL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TEXTINPUTH) $(TEXTINPUTL)

clean-$(MODNAME):
		@rm -f $(TEXTINPUTO) $(TEXTINPUTDO) $(LTEXTINPUTO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TEXTINPUTDEP) $(TEXTINPUTDS) $(TEXTINPUTDH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(TEXTINPUTO): CXXFLAGS += -I$(TEXTINPUTDIRS)
