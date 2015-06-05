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
                $(wildcard $(MODDIR)/lib/TagsExtension/*.cpp) \
                $(wildcard $(MODDIR)/lib/Utils/*.cpp)
CLINGO       := $(call stripsrc,$(CLINGS:.cpp=.o))
CLINGEXCEPO  := $(call stripsrc,$(MODDIR)/lib/Interpreter/RuntimeException.o)
CLINGCOMPDH  := $(call stripsrc,$(MODDIR)/lib/Interpreter/cling-compiledata.h)

CLINGDEP     := $(CLINGO:.o=.d)

CLINGETC_CLING := DynamicExprInfo.h DynamicLookupRuntimeUniverse.h \
        DynamicLookupLifetimeHandler.h Interpreter.h InvocationOptions.h \
        RuntimeUniverse.h Value.h RuntimeException.h

CLINGETC_LLVM := llvm/ADT/IntrusiveRefCntPtr.h \
        llvm/ADT/StringRef.h \
        llvm/ADT/SmallVector.h \
	llvm/ADT/iterator_range.h \
        llvm/Config/llvm-config.h \
        llvm/Support/AlignOf.h \
        llvm/Support/Allocator.h \
        llvm/Support/Casting.h \
        llvm/Support/Compiler.h \
        llvm/Support/DataTypes.h \
        llvm/Support/MathExtras.h \
        llvm/Support/Memory.h \
        llvm/Support/SwapByteOrder.h \
        llvm/Support/type_traits.h

CLINGETCPCH  := $(addprefix etc/cling/Interpreter/,$(CLINGETC_CLING)) \
	$(addprefix etc/cling/,$(CLINGETC_LLVM))
CLINGETC     := $(CLINGETCPCH) $(addprefix etc/cling/cint/,multimap multiset)

CLINGETC_ORIGINALS := $(addprefix $(call stripsrc,$(LLVMDIRI))/include/,$(CLINGETC_LLVM)) \
	$(addprefix $(CLINGDIR)/include/cling/,$(CLINGETC_CLING))

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
CLINGLLVMCXXFLAGS = $(patsubst -O%,,$(shell $(LLVMCONFIG) --cxxflags))
# -ffunction-sections breaks the debugger on some platforms ... and does not help libCling at all.
CLINGCXXFLAGS += -I$(CLINGDIR)/include $(filter-out -ffunction-sections,$(CLINGLLVMCXXFLAGS)) -fno-strict-aliasing

ifeq ($(CTORSINITARRAY),yes)
CLINGLDFLAGSEXTRA := -Wl,--no-ctors-in-init-array
endif

# Define NDEBUG for consistency with llvm and clang.
CLINGCXXNDEBUG := -DNDEBUG
ifeq ($(ROOTBUILD),debug)
  ifneq ($(LLVMDEV),)
    CLINGCXXNDEBUG := 
  endif
endif
CLINGCXXFLAGS += $(CLINGCXXNDEBUG)
CLINGCXXFLAGSNOI = $(patsubst -I%,,$(CLINGCXXFLAGS))

ifneq (,$(filter $(ARCH),win32gcc win64gcc))
# Hide llvm / clang symbols:
CLINGLDFLAGSEXTRA += -Wl,--exclude-libs,ALL 
endif

# used in $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) for not CLINGEXEO
CLINGEXCCXXFLAGS := -fno-exceptions
CLINGLIBEXTRA = $(CLINGLDFLAGSEXTRA) -L$(shell $(LLVMCONFIG) --libdir) \
	$(addprefix -lclang,\
		Frontend Serialization Driver CodeGen Parse Sema Analysis AST Edit Lex Basic) \
	$(shell $(LLVMCONFIG) --libs bitwriter orcjit mcjit native option ipo instrumentation objcarcopts profiledata)\
	$(shell $(LLVMCONFIG) --ldflags) $(shell $(LLVMCONFIG) --system-libs)

ifneq (,$(filter $(ARCH),win32gcc win64gcc))
# for EnumProcessModules() in TCling.cxx
CLINGLIBEXTRA += -lpsapi
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) FORCE

all-$(MODNAME):

clean-$(MODNAME):
		@rm -f $(CLINGO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLINGDEP) $(CLINGETC)

distclean::     distclean-$(MODNAME)

$(CLINGDIRS)/Module.mk: $(LLVMCONFIG)

$(CLINGETC_ORIGINALS): %: $(LLVMLIB)

etc/cling/llvm/%: $(call stripsrc,$(LLVMDIRI))/include/llvm/%
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp $< $@

etc/cling/cint/%: $(CLINGDIR)/include/cling/cint/%
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp $< $@

etc/cling/%.h: $(CLINGDIR)/include/cling/%.h
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp $< $@

$(CLINGDIR)/%.o: $(CLINGDIR)/%.cpp $(LLVMDEP)
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CXXMKDEPFLAGS) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) $(CXXOUT)$@ -c $<

$(call stripsrc,$(CLINGDIR)/%.o): $(CLINGDIR)/%.cpp $(LLVMDEP)
	$(MAKEDIR)
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS))  -D__cplusplus -- $<
	$(CXX) $(OPT) $(CXXMKDEPFLAGS) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) $(CXXOUT)$@ -c $<

$(CLINGCOMPDH): FORCE $(LLVMDEP)
	@mkdir -p $(dir $@)
	@echo '#define LLVM_CXX "$(CXX) $(OPT) $(CLINGCXXFLAGSNOI)"' > $@_tmp
	@diff -q $@_tmp $@ > /dev/null 2>&1 || mv $@_tmp $@
	@rm -f $@_tmp

ifneq ($(LLVMDEV),)
ifneq ($(PLATFORM),macosx)
# -Wl,-E exports all symbols, such that the JIT can find them.
# Doesn't exist on MacOS where this behavior is default.
CLINGLDEXPSYM := -Wl,-E
endif
$(CLINGEXE): $(CLINGO) $(CLINGEXEO) $(LTEXTINPUTO)
	$(RSYNC) --exclude '.svn' $(CLINGDIR) $(LLVMDIRO)/tools
	@cd $(LLVMDIRS)/tools && ln -sf ../../../cling # yikes
	@mkdir -p $(dir $@)
	$(LD) $(CLINGLDEXPSYM) -o $@ $(CLINGO) $(CLINGEXEO) $(LTEXTINPUTO) $(CLINGLIBEXTRA) 
endif

##### extra rules ######
ifneq ($(LLVMDEV),)
$(CLINGO)   : CLINGCXXFLAGS += '-DCLING_INCLUDE_PATHS="$(CLINGDIR)/include:$(shell pwd)/$(LLVMDIRO)/include:$(shell pwd)/$(LLVMDIRO)/tools/clang/include:$(LLVMDIRS)/include:$(LLVMDIRS)/tools/clang/include"'
$(CLINGEXEO): CLINGCXXFLAGS += -I$(TEXTINPUTDIRS)
$(CLINGEXEO): CLINGEXCCXXFLAGS := -fexceptions
else
endif

CLING_VERSION=ROOT_$(shell cat "$(CLINGDIR)/VERSION")

$(CLINGEXCEPO): CLINGEXCCXXFLAGS := -fexceptions
$(CLINGETC) : $(LLVMLIB)
$(CLINGO)   : $(CLINGETC)
$(call stripsrc,$(MODDIR)/lib/Interpreter/CIFactory.o): $(CLINGCOMPDH)
$(call stripsrc,$(MODDIR)/lib/Interpreter/CIFactory.o): CLINGCXXFLAGS += -I$(dir $(CLINGCOMPDH)) -pthread
$(call stripsrc,$(MODDIR)/lib/Interpreter/Interpreter.o): $(CLINGCOMPDH)
$(call stripsrc,$(MODDIR)/lib/Interpreter/Interpreter.o): CLINGCXXFLAGS += -I$(dir $(CLINGCOMPDH))
$(call stripsrc,$(MODDIR)/lib/Interpreter/Interpreter.o): CLINGCXXFLAGS += -DCLING_VERSION=$(CLING_VERSION)

