##===- cling/lib/UserInterface/Makefile---------------------*- Makefile -*-===##
# 
#                     The Cling Interpreter
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the a C++ interpreter user frontend.
#
##===----------------------------------------------------------------------===##

CLING_LEVEL = ../..
DIRS = textinput

LIBRARYNAME := clingUserInterface
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_SRC_DIR)/../../../clang/include \
-I$(PROJ_SRC_DIR)/../../../clang/lib \
-I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_OBJ_DIR)/../../../clang/include \
-I$(DESTDIR)$(PROJ_includedir)

include $(CLING_LEVEL)/Makefile

CXX.Flags := $(subst -fno-exceptions,,$(CXX.Flags))
