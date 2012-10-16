# Module.mk for cling module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2011-10-18

MODNAME      := cling
MODDIR       := $(ROOT_SRCDIR)/interpreter/$(MODNAME)

CLINGDIR     := $(MODDIR)

##### libCling #####
CLINGS       := $(wildcard $(MODDIR)/lib/Interpreter/*.cpp) \
                $(wildcard $(MODDIR)/lib/MetaProcessor/*.cpp) \
                $(wildcard $(MODDIR)/lib/Utils/*.cpp)
CLINGO       := $(call stripsrc,$(CLINGS:.cpp=.o))

CLINGDEP     := $(CLINGO:.o=.d)

CLINGETC     := $(addprefix etc/cling/Interpreter/,RuntimeUniverse.h ValuePrinter.h ValuePrinterInfo.h) \
                $(addprefix etc/cling/cint/,multimap multiset)

# used in the main Makefile
ALLHDRS      += $(CLINGETC)

ifneq ($(LLVMDEV),)
CLINGEXES    := $(wildcard $(MODDIR)/tools/driver/*.cpp) \
                $(wildcard $(MODDIR)/lib/UserInterface/*.cpp)
CLINGEXEO    := $(call stripsrc,$(CLINGEXES:.cpp=.o))
CLINGEXE     := $(LLVMDIRO)/Debug+Asserts/bin/cling
ALLEXECS     += $(CLINGEXE)
endif

# include all dependency files
INCLUDEFILES += $(CLINGDEP)

# include dir for picking up RuntimeUniverse.h etc - need to
# 1) copy relevant headers to include/
# 2) rely on TCling to addIncludePath instead of using CLING_..._INCL below
CLINGCXXFLAGS = $(patsubst -O%,,$(shell $(LLVMCONFIG) --cxxflags) -I$(CLINGDIR)/include \
	-fno-strict-aliasing)

ifeq ($(CTORSINITARRAY),yes)
CLINGLDFLAGSEXTRA := -Wl,--no-ctors-in-init-array
endif
CLINGLIBEXTRA = $(CLINGLDFLAGSEXTRA) -L$(shell $(LLVMCONFIG) --libdir) \
	$(addprefix -lclang,\
		Frontend Serialization Driver CodeGen Parse Sema Analysis RewriteCore AST Lex Basic Edit) \
	$(patsubst -lLLVM%Disassembler,,\
	$(filter-out -lLLVMipa,\
	$(shell $(LLVMCONFIG) --libs linker jit executionengine debuginfo \
	  archive bitreader all-targets codegen selectiondag asmprinter \
	  mcparser scalaropts instcombine transformutils analysis target))) \
	$(shell $(LLVMCONFIG) --ldflags)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

all-$(MODNAME):

clean-$(MODNAME):
		@rm -f $(CLINGO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLINGDEP) $(CLINGETC)

distclean::     distclean-$(MODNAME)

$(CLINGDIRS)/Module.mk: $(LLVMCONFIG)

etc/cling/%.h: $(CLINGDIR)/include/cling/%.h
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	@cp $< $@

etc/cling/%.h: $(call stripsrc,$(CLINGDIR)/%.o)/include/cling/%.h
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	@cp $< $@

etc/cling/cint/multimap: $(CLINGDIR)/include/cling/cint/multimap
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	@cp $< $@

etc/cling/cint/multiset: $(CLINGDIR)/include/cling/cint/multiset
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	@cp $< $@

$(CLINGDIR)/%.o: $(CLINGDIR)/%.cpp $(LLVMDEP)
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(CLINGCXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CLINGCXXFLAGS) $(CXXOUT)$@ -c $<

$(call stripsrc,$(CLINGDIR)/%.o): $(CLINGDIR)/%.cpp $(LLVMDEP)
	$(MAKEDIR)
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(CLINGCXXFLAGS)  -D__cplusplus -- $<
	$(CXX) $(OPT) $(CLINGCXXFLAGS) $(CXXOUT)$@ -c $<

ifneq ($(LLVMDEV),)
$(CLINGEXE): $(CLINGO) $(CLINGEXEO) $(LTEXTINPUTO)
	$(RSYNC) --exclude '.svn' $(CLINGDIR) $(LLVMDIRO)/tools
	@cd $(LLVMDIRS)/tools && ln -sf ../../../cling # yikes
	@mkdir -p $(dir $@)
	$(LD) $(CLINGLIBEXTRA) -o $@ $(CLINGO) $(CLINGEXEO) $(LTEXTINPUTO)
endif

##### extra rules ######
ifneq ($(LLVMDEV),)
$(CLINGO)   : CLINGCXXFLAGS += '-DCLING_SRCDIR_INCL="$(CLINGDIR)/include"' \
	'-DCLING_INSTDIR_INCL="$(shell cd $(LLVMDIRI); pwd)/include"'
$(CLINGEXEO): CLINGCXXFLAGS += -I$(TEXTINPUTDIRS)
endif
