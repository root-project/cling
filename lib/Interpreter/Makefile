##===- cling/lib/Interpreter/Makefile-----------------------*- Makefile -*-===##
# 
#                     The Cling Interpreter
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the a C++ interpreter backend.
#
##===----------------------------------------------------------------------===##

CLING_LEVEL = ../..
LIBRARYNAME := clingInterpreter
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

# Inform the sources about their install location
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_SRC_DIR)/../../../clang/include \
-I$(PROJ_SRC_DIR)/../../../clang/lib -I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_OBJ_DIR)/../../../clang/include \
-I$(DESTDIR)$(PROJ_includedir) \
'-DCLING_INCLUDE_PATHS="$(DESTDIR)$(PROJ_includedir):$(LLVM_SRC_ROOT)/tools/cling/include:$(LLVM_SRC_ROOT)/tools/clang/include:$(LLVM_SRC_ROOT)/include"'

include $(CLING_LEVEL)/Makefile

.PHONY: FORCE

$(ObjRootDir)/cling-compiledata.h: FORCE
	@mkdir -p $(dir $@)
	@echo '#define LLVM_CXX "$(CXX) $(CPP.BaseFlags) $(CXX.Flags)"' > $@_tmp
	@diff -q $@_tmp $@ > /dev/null 2>&1 || mv $@_tmp $@
	@rm -f $@_tmp

$(ObjRootDir)/RuntimeException.o: CXX.Flags := $(subst -fno-exceptions,,$(CXX.Flags))
$(ObjRootDir)/CIFactory.o: $(ObjRootDir)/cling-compiledata.h
$(ObjRootDir)/CIFactory.o: CXX.Flags += -I$(ObjRootDir) -pthread
$(ObjRootDir)/Interpreter.o: $(ObjRootDir)/cling-compiledata.h
$(ObjRootDir)/Interpreter.o: CXX.Flags += -I$(ObjRootDir)

