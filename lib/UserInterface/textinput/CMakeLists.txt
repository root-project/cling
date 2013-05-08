############################################################################
# CMakeLists.txt file for building ROOT core/textinout package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/textinput/*.cpp)

ROOT_GENERATE_DICTIONARY(G__TextInput *.h LINKDEF LinkDef.h)

set(TEXTINPUT_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__TextInput.cxx PARENT_SCOPE)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)
set(TEXTINPUT_CXX_SOURCES  ${sources} PARENT_SCOPE)
add_custom_target(TEXTINPUT_DICTIONARY DEPENDS G__TextInput.cxx)
set_target_properties(TEXTINPUT_DICTIONARY PROPERTIES FOLDER Dictionaries)
ROOT_INSTALL_HEADERS()
