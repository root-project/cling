//
// Created by Axel Naumann on 09/12/15.
//

//#include "cling/Interpreter/Jupyter/Kernel.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

#include <map>
#include <string>

namespace cling {
  namespace Jupyter {
    struct MIMEDataRef {
      const char* m_Data;
      const long m_Size;

      MIMEDataRef(const std::string& str):
      m_Data(str.c_str()), m_Size((long)str.length() + 1) {}
      MIMEDataRef(const char* str):
      MIMEDataRef(std::string(str)) {}
      MIMEDataRef(const char* data, long size):
      m_Data(data), m_Size(size) {}
    };

    /// Push MIME stuff to Jupyter. To be called from user code.
    ///\param contentDict - dictionary of MIME type versus content. E.g.
    /// {{"text/html", {"<div></div>", }}
    void pushOutput(const std::map<std::string, MIMEDataRef> contentDict) {
      FILE* outpipe = popen("???", "w");

      // Pipe sees (all numbers are longs, except for the first:
      // - num bytes in a long (sent as a single unsigned char!)
      // - num elements of the MIME dictionary; Jupyter selects one to display.
      // For each MIME dictionary element:
      //   - MIME type as 0-terminated string
      //   - size of MIME data buffer (including the terminating 0 for
      //     0-terminated strings)
      //   - MIME data buffer

      // Write number of dictionary elements (and the size of that number in a
      // char)
      unsigned char sizeLong = sizeof(long);
      fwrite(&sizeLong, 1, 1, outpipe);
      long dictSize = contentDict.size();
      fwrite(&dictSize, sizeof(long), 1, outpipe);

      for (auto iContent: contentDict) {
        const std::string& mimeType = iContent.first;
        fwrite(mimeType.c_str(), mimeType.size() + 1, 1, outpipe);
        const MIMEDataRef& mimeData = iContent.second;
        fwrite(&mimeData.m_Size, sizeof(long), 1, outpipe);
        fwrite(mimeData.m_Data, mimeData.m_Size, 1, outpipe);
      }

      pclose(outpipe);
    }
  } // namespace Jupyter
} // namespace cling

extern "C" {
///\{
///\name Cling4CTypes
/// The Python compatible view of cling

/// The Interpreter object cast to void*
using TheInterpreter = void ;

/// Create an interpreter object.
TheInterpreter*
cling_create(int argc, const char *argv[], const char* llvmdir) {
  auto interp = new cling::Interpreter(argc, argv, llvmdir);
  return interp;
}


/// Destroy the interpreter.
void cling_destroy(TheInterpreter *interpVP) {
  cling::Interpreter *interp = (cling::Interpreter *) interpVP;
  delete interp;
}


/// Evaluate a string of code. Returns 0 on success.
int cling_eval(TheInterpreter *interpVP, const char *code) {
  cling::Interpreter *interp = (cling::Interpreter *) interpVP;
  //cling::Value V;
  cling::Interpreter::CompilationResult Res = interp->process(code /*, V*/);
  if (Res != cling::Interpreter::kSuccess)
    return 1;
  cling::Jupyter::pushOutput({{"text/html", "You just executed C++ code!"}});
  return 0;
}

///\}

} // extern "C"
