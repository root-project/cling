#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set(SOURCES
    Kernel.cpp
    )

set(LIBS
    clangBasic
    clangAST
    clangFrontend
    clingInterpreter
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

add_clang_library(libclingJupyter ${ENABLE_SHARED} ${ENABLE_STATIC}
        OUTPUT_NAME ${output_name}
        ${SOURCES}
#        DEPENDS clang-headers

        LINK_LIBS
        ${LIBS}

        LINK_COMPONENTS
        Core
        Support
        )

if(ENABLE_SHARED)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(LIBCLINGJUPYTER_LINK_FLAGS " -Wl,-compatibility_version -Wl,1")
        if (DEFINED ${LLVM_SUBMIT_VERSION})
            set(LIBCLINGJUPYTER_LINK_FLAGS
                    "${LIBCLINGJUPYTER_LINK_FLAGS} -Wl,-current_version -Wl,${LLVM_SUBMIT_VERSION}.${LLVM_SUBMIT_SUBVERSION}")
        endif()

        set_property(TARGET libclingJupyter APPEND_STRING PROPERTY
                LINK_FLAGS ${LIBCLINGJUPYTER_LINK_FLAGS})
    endif()
endif()
