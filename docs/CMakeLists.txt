
if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_builddir ${CMAKE_CURRENT_BINARY_DIR})

  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()

  if (LLVM_DOXYGEN_EXTERNAL_SEARCH)
    set(enable_searchengine "YES")
    set(searchengine_url "${LLVM_DOXYGEN_SEARCHENGINE_URL}")
    set(enable_server_based_search "YES")
    set(enable_external_search "YES")
    set(extra_search_mappings "${LLVM_DOXYGEN_SEARCH_MAPPINGS}")
  else()
    set(enable_searchengine "NO")
    set(searchengine_url "")
    set(enable_server_based_search "NO")
    set(enable_external_search "NO")
    set(extra_search_mappings "")
  endif()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)
  set(enable_searchengine)
  set(searchengine_url)
  set(enable_server_based_search)
  set(enable_external_search)
  set(extra_search_mappings)
  set(cling_doxygen_generate_qhp)
  set(cling_doxygen_qch_filename)
  set(cling_doxygen_qhp_namespace)
  set(cling_doxygen_qhelpgenerator_path)
  set(cling_doxygen_qhp_cust_filter_name)
  set(cling_doxygen_qhp_cust_filter_attrs)

  add_custom_target(doxygen-cling
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating cling doxygen documentation." VERBATIM)

  if (LLVM_BUILD_DOCS)
    add_dependencies(doxygen doxygen-cling)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()

if (LLVM_ENABLE_SPHINX)
  if (SPHINX_FOUND)
    include(AddSphinxTarget)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html cling)
    endif()
  endif()
endif()
