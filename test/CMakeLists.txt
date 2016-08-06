#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

# Needed for '%shlibext'
set(TARGET_SHLIBEXT "${CMAKE_SHARED_LIBRARY_SUFFIX}")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

option(CLING_TEST_USE_VG "Run Cling tests under Valgrind" OFF)
if(CLING_TEST_USE_VG)
  set(CLING_TEST_EXTRA_ARGS ${CLING_TEST_EXTRA_ARGS} "--vg")
endif ()

list(APPEND CLING_TEST_DEPS cling llvm-config FileCheck clang c-index-test
                            clang-check clang-format opt count not)

set(CLING_TEST_PARAMS
  cling_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_custom_target(cling-test-depends DEPENDS ${CLING_TEST_DEPS})

add_lit_testsuite(check-cling "Running the Cling regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  PARAMS ${CLING_TEST_PARAMS}
  DEPENDS ${CLING_TEST_DEPS}
  ARGS ${CLING_TEST_EXTRA_ARGS}
  )
set_target_properties(check-cling PROPERTIES FOLDER "Cling tests")

# Add a legacy target spelling: cling-test
add_custom_target(cling-test)
add_dependencies(cling-test check-cling)
set_target_properties(cling-test PROPERTIES FOLDER "Cling tests")
