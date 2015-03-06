##===- Makefile --------------------------------------------*- Makefile -*-===##
#
#                     The Cling Interpreter
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# If CLING_LEVEL is not set, then we are the top-level Makefile. Otherwise, we
# are being included from a subdirectory makefile.

ifndef CLING_LEVEL

IS_TOP_LEVEL := 1
CLING_LEVEL := .
DIRS := include lib tools docs

PARALLEL_DIRS :=

ifeq ($(BUILD_EXAMPLES),1)
  PARALLEL_DIRS += examples
endif
endif

ifeq ($(MAKECMDGOALS),libs-only)
  DIRS := $(filter-out tools docs, $(DIRS))
  OPTIONAL_DIRS :=
endif

###
# Common Makefile code, shared by all Cling Makefiles.

# Set LLVM source root level.
LEVEL := $(CLING_LEVEL)/../..

# Include LLVM common makefile.
include $(LEVEL)/Makefile.common

ifneq ($(ENABLE_DOCS),1)
  DIRS := $(filter-out docs, $(DIRS))
endif

# Set common Cling build flags.
CPP.Flags += -I$(PROJ_SRC_DIR)/$(CLING_LEVEL)/include -I$(PROJ_OBJ_DIR)/$(CLING_LEVEL)/include
ifdef CLING_VENDOR
CPP.Flags += -DCLING_VENDOR='"$(CLING_VENDOR) "'
endif

# Disable -fstrict-aliasing. Darwin disables it by default (and LLVM doesn't
# work with it enabled with GCC), Cling/llvm-gc don't support it yet, and newer
# GCC's have false positive warnings with it on Linux (which prove a pain to
# fix). For example:
#   http://gcc.gnu.org/PR41874
#   http://gcc.gnu.org/PR41838
#
# We can revisit this when LLVM/Cling support it.
CXX.Flags += -fno-strict-aliasing

# Determine cling's version:
CLING_VERSION=$(shell cat $(PROJ_SRC_DIR)/$(CLING_LEVEL)/VERSION)
CPP.Flags += -DCLING_VERSION='$(CLING_VERSION)'

###
# Cling Top Level specific stuff.

ifeq ($(IS_TOP_LEVEL),1)

ifneq ($(PROJ_SRC_ROOT),$(PROJ_OBJ_ROOT))
$(RecursiveTargets)::
	$(Verb) if [ ! -f test/Makefile ]; then \
	  $(MKDIR) test; \
	  $(CP) $(PROJ_SRC_DIR)/test/Makefile test/Makefile; \
	fi
endif

test::
	@ $(MAKE) -C test

report::
	@ $(MAKE) -C test report

clean::
	@ $(MAKE) -C test clean

libs-only: all

tags::
	$(Verb) etags `find . -type f -name '*.h' -or -name '*.cpp' | \
	  grep -v /lib/Headers | grep -v /test/`

cscope.files:
	find tools lib include -name '*.cpp' \
	                    -or -name '*.def' \
	                    -or -name '*.td' \
	                    -or -name '*.h' > cscope.files

.PHONY: test report clean cscope.files

endif
