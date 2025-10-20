#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

# Keep symbols for JIT resolution
set(LLVM_NO_DEAD_STRIP 1)

set(SOURCES
    Kernel.cpp
    )
if(MSVC)
    set_source_files_properties(Kernel.cpp COMPILE_FLAGS "/EHsc /GR")
else()
    set_source_files_properties(Kernel.cpp COMPILE_FLAGS "-fexceptions -frtti")
endif()
#Solve unresolved symbols bug in unix
#See https://github.com/vgvassilev/cling/issues/114
if(WIN32)
   #FIXME: I don't know what flags are used in windows
   #to include all symbols from a static library
   #Maybe the bug is not present in windows and we leave it 
   #as it is
   set(INTERPRETER
      clingInterpreter
      )
else()
   #Force all interpreter symbols to be present in the shared library
   #this will prevent missing symbol errors because we don't know at
   #link time what function calls will be made by the user of 
   #the shared library
   if (APPLE)
      set(INTERPRETER
         -Wl,-force_load clingInterpreter 
         )
   else()
      set(INTERPRETER
         -Wl,--whole-archive clingInterpreter -Wl,--no-whole-archive
         )
   endif()
endif()

set(LIBS
        clangAST
        clangBasic
        clangCodeGen
        clangDriver
        clangFrontend
        clangLex
        clangParse
        clangSema
        clangSerialization

        clingUserInterface
        clingMetaProcessor
        ${INTERPRETER}
        clingUtils
    )

if( LLVM_ENABLE_PIC )
    set(ENABLE_SHARED SHARED)
endif()

if(WIN32)
    set(output_name "libclingJupyter")
else()
    set(output_name "clingJupyter")
endif()

add_cling_library(libclingJupyter ${ENABLE_SHARED} ${ENABLE_STATIC}
        OUTPUT_NAME ${output_name}
        ${SOURCES}

        LINK_LIBS
        ${LIBS}

        LINK_COMPONENTS
        Core
        Support
        )

set_target_properties(libclingJupyter
        PROPERTIES ENABLE_EXPORTS 1)

if(MSVC)
    set_target_properties(libclingJupyter PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)

    # RTTI/C++ symbols
    set(jupyter_exports ${jupyter_exports} ??_7type_info@@6B@
        ?__type_info_root_node@@3U__type_info_node@@A
        ?nothrow@std@@3Unothrow_t@1@B
        __std_find_trivial_1
        __std_reverse_trivially_swappable_1
    )

    # Compiler added symbols for static variables. NOT for VStudio < 2015
    set(jupyter_exports ${jupyter_exports} _Init_thread_abort _Init_thread_epoch
        _Init_thread_footer _Init_thread_header _tls_index
    )

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # new/delete variants needed when linking to static msvc runtime (esp. Debug)
        set(jupyter_exports ${jupyter_exports}
        ??2@YAPEAX_K@Z
        ??3@YAXPEAX@Z
        ??_U@YAPEAX_K@Z
        ??_V@YAXPEAX@Z
        ??3@YAXPEAX_K@Z
        ??2@YAPEAX_KAEBUnothrow_t@std@@@Z
        ??_U@YAPEAX_KAEBUnothrow_t@std@@@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@M@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@N@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z
        ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z
        ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z
        ?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z
        )
    else()
        set(jupyter_exports ${jupyter_exports}
        ??2@YAPAXI@Z
        ??3@YAXPAX@Z
        ??3@YAXPAXI@Z
        ??_U@YAPAXI@Z
        ??_V@YAXPAX@Z
        ??_V@YAXPAXI@Z
        ??2@YAPAXIABUnothrow_t@std@@@Z
        ??_U@YAPAXIABUnothrow_t@std@@@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@H@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@M@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@N@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@PBX@Z
        ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@P6AAAV01@AAV01@@Z@Z
        ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@D@Z
        ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@PBD@Z
        ?_Facet_Register@std@@YAXPAV_Facet_base@1@@Z
        )
    endif()

    # List to '/EXPORT:sym0 /EXPORT:sym1 /EXPORT:sym2 ...'
    foreach(sym ${jupyter_exports})
        set(cling_link_str "${cling_link_str} /EXPORT:${sym}")
    endforeach(sym ${jupyter_exports})

    set_property(TARGET libclingJupyter APPEND_STRING PROPERTY LINK_FLAGS ${cling_link_str})

endif(MSVC)

if(ENABLE_SHARED)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(LIBCLINGJUPYTER_LINK_FLAGS " -Wl,-compatibility_version -Wl,1")
        if (DEFINED LLVM_SUBMIT_VERSION)
            set(LIBCLINGJUPYTER_LINK_FLAGS
                    "${LIBCLINGJUPYTER_LINK_FLAGS} -Wl,-current_version -Wl,${LLVM_SUBMIT_VERSION}.${LLVM_SUBMIT_SUBVERSION}")
        endif()

        set_property(TARGET libclingJupyter APPEND_STRING PROPERTY
                LINK_FLAGS ${LIBCLINGJUPYTER_LINK_FLAGS})
    endif()
endif()
