# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(CLING_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/cling)
set(cling_cmake_builddir "${CMAKE_BINARY_DIR}/${CLING_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(CLING_EXPORTS GLOBAL PROPERTY CLING_EXPORTS)
export(TARGETS ${CLING_EXPORTS} FILE ${cling_cmake_builddir}/ClingTargets.cmake)

# Generate ClingConfig.cmake for the build tree.
set(CLING_CONFIG_CMAKE_DIR "${cling_cmake_builddir}")
set(CLING_CONFIG_CLANG_CMAKE_DIR "${clang_cmake_builddir}")
set(CLING_CONFIG_EXPORTS_FILE "${cling_cmake_builddir}/ClingTargets.cmake")
set(CLING_CONFIG_INCLUDE_DIRS
  "${CLING_SOURCE_DIR}/include"
  "${CLING_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClingConfig.cmake.in
  ${cling_cmake_builddir}/ClingConfig.cmake
  @ONLY)
set(CLING_CONFIG_CMAKE_DIR)
set(CLING_CONFIG_LLVM_CMAKE_DIR)
set(CLING_CONFIG_EXPORTS_FILE)

# Generate ClingConfig.cmake for the install tree.
set(CLING_CONFIG_CODE "
# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(CLING_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${CLING_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(CLING_CONFIG_CODE "${CLING_CONFIG_CODE}
get_filename_component(CLING_INSTALL_PREFIX \"\${CLING_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(CLING_CONFIG_CMAKE_DIR "\${CLING_INSTALL_PREFIX}/${CLING_INSTALL_PACKAGE_DIR}")
set(CLING_CONFIG_LLVM_CMAKE_DIR "\${CLING_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(CLING_CONFIG_EXPORTS_FILE "\${CLING_CMAKE_DIR}/ClingTargets.cmake")
set(CLING_CONFIG_INCLUDE_DIRS
  "\${CLING_INSTALL_PREFIX}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClingConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClingConfig.cmake
  @ONLY)
set(CLING_CONFIG_CODE)
set(CLING_CONFIG_CMAKE_DIR)
set(CLING_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  get_property(cling_has_exports GLOBAL PROPERTY CLING_HAS_EXPORTS)
  if(cling_has_exports)
    install(EXPORT ClingTargets DESTINATION ${CLING_INSTALL_PACKAGE_DIR}
            COMPONENT cling-cmake-exports)
  endif()

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClingConfig.cmake
    DESTINATION ${CLING_INSTALL_PACKAGE_DIR}
    COMPONENT cling-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(cling-cmake-exports)
    add_llvm_install_targets(install-cling-cmake-exports
                             COMPONENT cling-cmake-exports)
  endif()
endif()
