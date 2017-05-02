//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// FIXME: This file shall contain the decl of cling::Jupyter in a future
// revision!
//#include "cling/Interpreter/Jupyter/Kernel.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/Utils/Output.h"

#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <cstring>
#ifndef LLVM_ON_WIN32
# include <unistd.h>
#else
# include <io.h>
# define write _write
#endif

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
    ///\returns `false` if the output could not be sent.
    bool pushOutput(const std::map<std::string, MIMEDataRef> contentDict) {

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
      if (write(pipeToJupyterFD, &sizeLong, 1) != 1)
        return false;
      long dictSize = contentDict.size();
      if (write(pipeToJupyterFD, &dictSize, sizeof(long)) != sizeof(long))
        return false;

      for (auto iContent: contentDict) {
        const std::string& mimeType = iContent.first;
        long mimeTypeSize = (long)mimeType.size();
        if (write(pipeToJupyterFD, &mimeTypeSize, sizeof(long)) != sizeof(long))
          return false;
        if (write(pipeToJupyterFD, mimeType.c_str(), mimeType.size() + 1)
            != (long)(mimeType.size() + 1))
          return false;
        const MIMEDataRef& mimeData = iContent.second;
        if (write(pipeToJupyterFD, &mimeData.m_Size, sizeof(long))
            != sizeof(long))
          return false;
        if (write(pipeToJupyterFD, mimeData.m_Data, mimeData.m_Size)
            != mimeData.m_Size)
          return false;
      }
      return true;
    }
  } // namespace Jupyter
} // namespace cling

extern "C" {
///\{
///\name Cling4CTypes
/// The Python compatible view of cling

/// The MetaProcessor cast to void*
using TheMetaProcessor = void;

/// Create an interpreter object.
TheMetaProcessor*
cling_create(int argc, const char *argv[], const char* llvmdir, int pipefd) {
  pipeToJupyterFD = pipefd;
  auto I = new cling::Interpreter(argc, argv, llvmdir);
  return new cling::MetaProcessor(*I, cling::errs());
}


/// Destroy the interpreter.
void cling_destroy(TheMetaProcessor *metaProc) {
  cling::MetaProcessor *M = (cling::MetaProcessor*)metaProc;
  cling::Interpreter *I = const_cast<cling::Interpreter*>(&M->getInterpreter());
  delete M;
  delete I;
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
char* cling_eval(TheMetaProcessor *metaProc, const char *code) {
  cling::MetaProcessor *M = (cling::MetaProcessor*)metaProc;
  cling::Value V;
  cling::Interpreter::CompilationResult Res;
  if (M->process(code, Res, &V, /*disableValuePrinting*/ true)) {
    cling::Jupyter::pushOutput({{"text/html", "Incomplete input! Ignored."}});
    M->cancelContinuation();
    return nullptr;
  }
  if (Res != cling::Interpreter::kSuccess)
    return nullptr;

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
