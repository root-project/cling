set(CLING_TEST_DIRECTORIES
  "ErrorRecovery"
  "Extensions"
  "LibraryCall"
  "Prompt"
  "Recursion"
  "SourceCall")

set(LLVM_SOURCE_DIR "${LLVM_MAIN_SRC_DIR}")
set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_BUILD_MODE "%(build_mode)s")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/%(build_config)s")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/%(build_config)s")
set(CLING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLING_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")
if(BUILD_SHARED_LIBS)
  set(ENABLE_SHARED 1)
else()
  set(ENABLE_SHARED 0)
endif(BUILD_SHARED_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

#configure_file(
#  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
#  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
#  @ONLY)

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  if( LLVM_MAIN_SRC_DIR )
    set(LIT "${LLVM_SOURCE_DIR}/utils/lit/lit.py")
  else()
    set(LIT "${PATH_TO_LLVM_BUILD}/bin/${CMAKE_CFG_INTDIR}/llvm-lit")
    # Installed LLVM does not contain ${CMAKE_CFG_INTDIR} in paths.
    if( NOT EXISTS ${LIT} )
      set(LIT "${PATH_TO_LLVM_BUILD}/bin/llvm-lit")
    endif()
  endif()

  if( PATH_TO_LLVM_BUILD )
    set(CLING_TEST_EXTRA_ARGS "--path=${CLING_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}")
  endif()

  option(CLING_TEST_USE_VG "Run Cling tests under Valgrind" OFF)
  if(CLING_TEST_USE_VG)
    set(CLING_TEST_EXTRA_ARGS ${CLING_TEST_EXTRA_ARGS} "--vg")
  endif ()

  set(LIT_ARGS "${CLING_TEST_EXTRA_ARGS} ${LLVM_LIT_ARGS}")
  separate_arguments(LIT_ARGS)

  add_custom_target(cling-test.deps)
  set_target_properties(cling-test.deps PROPERTIES FOLDER "Cling tests")

  add_custom_target(cling-test
    COMMAND ${PYTHON_EXECUTABLE}
                ${LIT}
                --param claig_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                --param cling_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
                --param build_config=${CMAKE_CFG_INTDIR}
                --param build_mode=${RUNTIME_BUILD_MODE}
                ${LIT_ARGS}
                ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running Cling regression tests")

#  if( NOT CLING_BUILT_STANDALONE )
#    add_custom_target(check-all
#      COMMAND ${PYTHON_EXECUTABLE}
#                ${LIT}
#                --param build_config=${CMAKE_CFG_INTDIR}
#                --param build_mode=${RUNTIME_BUILD_MODE}
#                ${LIT_ARGS}
#                ${LLVM_BINARY_DIR}/test
#                ${CMAKE_CURRENT_BINARY_DIR}
#                COMMENT "Running Cling and LLVM regression tests")
#    add_dependencies(check-all cling-test.deps)
#    if ( LLVM_INCLUDE_TESTS )
#      add_dependencies(cling-test.deps ClangUnitTests)
#      add_dependencies(check-all check.deps)
#    endif ( LLVM_INCLUDE_TESTS )
#    add_dependencies(cling-test.deps
#                     llvm-dis llc opt
#                     FileCheck count not
#                     )
#  endif()

  add_dependencies(cling-test cling-test.deps)
  add_dependencies(cling-test.deps
                 cling cling-headers c-index-test diagtool arcmt-test c-arcmt-test
                 )

endif()
