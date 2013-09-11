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
CLINGEXCEPO  := $(call stripsrc,$(MODDIR)/lib/Interpreter/RuntimeException.o)

CLINGDEP     := $(CLINGO:.o=.d)

CLINGETC_CLING := DynamicExprInfo.h DynamicLookupRuntimeUniverse.h \
        DynamicLookupLifetimeHandler.h Interpreter.h InvocationOptions.h \
        RuntimeUniverse.h StoredValueRef.h Value.h \
        ValuePrinter.h ValuePrinterInfo.h RuntimeException.h

CLINGETC_LLVM := llvm/ADT/IntrusiveRefCntPtr.h \
        llvm/ADT/OwningPtr.h \
        llvm/ADT/StringRef.h \
        llvm/Config/llvm-config.h \
        llvm/Support/Casting.h \
        llvm/Support/Compiler.h \
        llvm/Support/DataTypes.h \
        llvm/Support/type_traits.h

CLINGETC     := $(addprefix etc/cling/Interpreter/,$(CLINGETC_CLING)) \
        $(addprefix etc/cling/cint/,multimap multiset) \
	$(addprefix etc/cling/,$(CLINGETC_LLVM))

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
CLINGCXXFLAGS = -I$(CLINGDIR)/include $(patsubst -O%,,$(shell $(LLVMCONFIG) --cxxflags) \
	-fno-strict-aliasing)

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

ifeq ($(ARCH),win32gcc)
# Hide llvm / clang symbols:
CLINGLDFLAGSEXTRA += -Wl,--exclude-libs,ALL 
endif

# used in $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) for not CLINGEXEO
CLINGEXCCXXFLAGS := -fno-exceptions
CLINGLIBEXTRA = $(CLINGLDFLAGSEXTRA) -L$(shell $(LLVMCONFIG) --libdir) \
	$(addprefix -lclang,\
		Frontend Serialization Driver CodeGen Parse Sema Analysis RewriteCore AST Edit Lex Basic) \
	$(patsubst -lLLVM%Disassembler,,\
	$(filter-out -lLLVMipa,$(shell $(LLVMCONFIG) --libs)))\
	$(shell $(LLVMCONFIG) --ldflags)

ifeq ($(ARCH),win32gcc)
# for EnumProcessModules() in TCling.cxx
CLINGLIBEXTRA += -lpsapi
endif

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
	$(CXX) $(OPT) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) $(CXXOUT)$@ -c $<

$(call stripsrc,$(CLINGDIR)/%.o): $(CLINGDIR)/%.cpp $(LLVMDEP)
	$(MAKEDIR)
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS))  -D__cplusplus -- $<
	$(CXX) $(OPT) $(subst -fno-exceptions,$(CLINGEXCCXXFLAGS),$(CLINGCXXFLAGS)) $(CXXOUT)$@ -c $<

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
$(CLINGO)   : CLINGCXXFLAGS += '-DCLING_SRCDIR_INCL="$(CLINGDIR)/include"' \
	'-DCLING_INSTDIR_INCL="$(shell cd $(LLVMDIRI); pwd)/include"'
$(CLINGEXEO): CLINGCXXFLAGS += -I$(TEXTINPUTDIRS)
$(CLINGEXEO): CLINGEXCCXXFLAGS := -fexceptions
else
endif

$(CLINGEXCEPO): CLINGEXCCXXFLAGS := -fexceptions
$(CLINGETC) : $(LLVMLIB)
$(CLINGO)   : $(CLINGETC)
