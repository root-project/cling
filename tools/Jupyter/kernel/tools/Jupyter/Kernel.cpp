//
// Created by Axel Naumann on 09/12/15.
//

//#include "cling/Interpreter/Jupyter/Kernel.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <cstring>

// FIXME: should be moved into a Jupyter interp struct that then gets returned
// from create.
int pipeToJupyterFD = -1;

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

      // Pipe sees (all numbers are longs, except for the first:
      // - num bytes in a long (sent as a single unsigned char!)
      // - num elements of the MIME dictionary; Jupyter selects one to display.
      // For each MIME dictionary element:
      //   - size of MIME type string  (including the terminating 0)
      //   - MIME type as 0-terminated string
      //   - size of MIME data buffer (including the terminating 0 for
      //     0-terminated strings)
      //   - MIME data buffer

      // Write number of dictionary elements (and the size of that number in a
      // char)
      unsigned char sizeLong = sizeof(long);
      write(pipeToJupyterFD, &sizeLong, 1);
      long dictSize = contentDict.size();
      write(pipeToJupyterFD, &dictSize, sizeof(long));

      for (auto iContent: contentDict) {
        const std::string& mimeType = iContent.first;
        long mimeTypeSize = (long)mimeType.size();
        write(pipeToJupyterFD, &mimeTypeSize, sizeof(long));
        write(pipeToJupyterFD, mimeType.c_str(), mimeType.size() + 1);
        const MIMEDataRef& mimeData = iContent.second;
        write(pipeToJupyterFD, &mimeData.m_Size, sizeof(long));
        write(pipeToJupyterFD, mimeData.m_Data, mimeData.m_Size);
      }
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
cling_create(int argc, const char *argv[], const char* llvmdir, int pipefd) {
  auto interp = new cling::Interpreter(argc, argv, llvmdir);
  pipeToJupyterFD = pipefd;
  return interp;
}


/// Destroy the interpreter.
void cling_destroy(TheInterpreter *interpVP) {
  cling::Interpreter *interp = (cling::Interpreter *) interpVP;
  delete interp;
}

/// Stringify a cling::Value
static std::string ValueToString(const cling::Value& V) {
  std::string valueString;
  {
    llvm::raw_string_ostream os(valueString);
    V.print(os);
  }
  return valueString;
}

/// Evaluate a string of code. Returns nullptr on failure.
/// Returns a string representation of the expression (can be "") on success.
char* cling_eval(TheInterpreter *interpVP, const char *code) {
  cling::Interpreter *interp = (cling::Interpreter *) interpVP;
  cling::Value V;
  cling::Interpreter::CompilationResult Res = interp->evaluate(code, V);
  if (Res != cling::Interpreter::kSuccess)
    return nullptr;

  cling::Jupyter::pushOutput({{"text/html", "You just executed C++ code!"}});
  if (!V.isValid())
    return strdup("");
  return strdup(ValueToString(V).c_str());
}

void cling_eval_free(char* str) {
  free(str);
}

/// Code completion interfaces.

/// Start completion of code. Returns a handle to be passed to
/// cling_complete_next() to iterate over the completion options. Returns nulptr
/// if no completions are known.
void* cling_complete_start(const char* code) {
  return new int(42);
}

/// Grab the next completion of some code. Returns nullptr if none is left.
const char* cling_complete_next(void* completionHandle) {
  int* counter = (int*) completionHandle;
  if (++(*counter) > 43) {
    delete counter;
    return nullptr;
  }
  return "COMPLETE!";
}

///\}

} // extern "C"
