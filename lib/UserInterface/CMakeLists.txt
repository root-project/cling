#set(LLVM_LINK_COMPONENTS mc)
#set(LLVM_USED_LIBS clangBasic clangAST clangParse)


add_cling_library(clingUserInterface
  UserInterface.cpp
  textinput/Editor.cpp
  textinput/History.cpp
  textinput/KeyBinding.cpp
  textinput/Range.cpp
  textinput/SignalHandler.cpp
  textinput/StreamReader.cpp
  textinput/StreamReaderUnix.cpp
  textinput/StreamReaderWin.cpp
  textinput/TerminalConfigUnix.cpp
  textinput/TerminalDisplay.cpp
  textinput/TerminalDisplayUnix.cpp
  textinput/TerminalDisplayWin.cpp
  textinput/TextInput.cpp
  textinput/TextInputContext.cpp
)

#Find svn information
EXECUTE_PROCESS(COMMAND ${Subversion_SVN_EXECUTABLE} info "${CMAKE_CURRENT_LIST_DIR}" RESULT_VARIABLE svn_result OUTPUT_VARIABLE svn_ver)
IF (NOT ${svn_result} EQUAL 0) 
	MESSAGE(SEND_ERROR "Svn command returned error, is it installed?")
ELSE(NOT ${svn_result} EQUAL 0) 

	#Find and clean the url of the actual repository
	STRING(REGEX MATCH  "URL: ([^\n]+)"  url "${svn_ver}")
	STRING(REPLACE      "URL: " "" url "${url}")

	#Point the url to the rigt location
	STRING(REPLACE  "interpreter/cling/lib/UserInterface" "core/textinput/src/textinput" url "${url}")

	#Checkout the source
	EXECUTE_PROCESS(COMMAND ${Subversion_SVN_EXECUTABLE} co "${url}" "${CMAKE_CURRENT_LIST_DIR}/textinput" RESULT_VARIABLE scn_co_result)
	IF (NOT ${scn_co_result} EQUAL 0) 
		MESSAGE(SEND_ERROR "Svn checkout for ${url} error")
	ENDIF (NOT ${scn_co_result} EQUAL 0) 
ENDIF(NOT ${svn_result} EQUAL 0) 
	
  if(MSVC)
    # windows.h doesn't compile with /Za
    get_target_property(NON_ANSI_COMPILE_FLAGS clingUserInterface COMPILE_FLAGS)
    string(REPLACE "/Za" "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
    set_target_properties(clingUserInterface PROPERTIES
      COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  endif()

#add_subdirectory(textinput)
#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)
