#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------

add_cling_library(clingDemoPlugin SHARED DemoPlugin.cpp)
if (TARGET check-cling)
  # Make sure if we are running the testsuite to build clingDemoPlugin before.
  add_dependencies(check-cling clingDemoPlugin)
endif()

set_target_properties(clingDemoPlugin PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".")
if(APPLE)
  target_link_libraries(clingDemoPlugin PUBLIC -Wl,-bind_at_load -Wl,-undefined -Wl,dynamic_lookup)
elseif(NOT MSVC)
  target_link_libraries(clingDemoPlugin PUBLIC -Wl,--unresolved-symbols=ignore-in-object-files)
endif()

