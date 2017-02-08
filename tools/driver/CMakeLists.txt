#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

# Keep symbols for JIT resolution
set(LLVM_NO_DEAD_STRIP 1)

if(BUILD_SHARED_LIBS)
  set(LIBS
    LLVMSupport
    clangFrontendTool
    clingInterpreter
    clingMetaProcessor
    clingUserInterface
    clingUtils
  )
  add_cling_executable(cling
    cling.cpp
  )
else()
  set(LIBS
    LLVMSupport
    clangFrontendTool
    clingUserInterface
  )
  add_cling_executable(cling
    cling.cpp
    $<TARGET_OBJECTS:obj.clingInterpreter>
    $<TARGET_OBJECTS:obj.clingMetaProcessor>
    $<TARGET_OBJECTS:obj.clingUtils>
  )
endif(BUILD_SHARED_LIBS)

set_target_properties(cling
  PROPERTIES ENABLE_EXPORTS 1)

if(MSVC)
  set_target_properties(cling
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
  set_property(TARGET cling APPEND_STRING PROPERTY LINK_FLAGS
    " /EXPORT:??_7type_info@@6B@ /EXPORT:?__type_info_root_node@@3U__type_info_node@@A /EXPORT:_Init_thread_abort /EXPORT:_Init_thread_epoch /EXPORT:_Init_thread_footer /EXPORT:_Init_thread_header /EXPORT:_tls_index /EXPORT:?getAsString@QualType@clang@@SA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBVType@2@VQualifiers@2@@Z /EXPORT:?print@Decl@clang@@QEBAXAEAVraw_ostream@llvm@@I_N@Z /EXPORT:??6raw_ostream@llvm@@QEAAAEAV01@PEBX@Z /EXPORT:?getQualifiedNameAsString@NamedDecl@clang@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@XZ /EXPORT:?insert_imp_big@SmallPtrSetImplBase@llvm@@AEAA?AU?$pair@PEBQEBX_N@std@@PEBX@Z /EXPORT:?getAsStringInternal@QualType@clang@@SAXPEBVType@2@VQualifiers@2@AEAV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBUPrintingPolicy@2@@Z /EXPORT:?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z /EXPORT:?decls_begin@DeclContext@clang@@QEBA?AVdecl_iterator@12@XZ /EXPORT:?nothrow@std@@3Unothrow_t@1@B /EXPORT:?field_begin@RecordDecl@clang@@QEBA?AV?$specific_decl_iterator@VFieldDecl@clang@@@DeclContext@2@XZ /EXPORT:?errs@llvm@@YAAEAVraw_ostream@1@XZ /EXPORT:?grow_pod@SmallVectorBase@llvm@@IEAAXPEAX_K1@Z /EXPORT:?write@raw_ostream@llvm@@QEAAAEAV12@E@Z")
endif(MSVC)

target_link_libraries(cling
  ${LIBS}
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
