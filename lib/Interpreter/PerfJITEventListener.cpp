//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Guilherme Amadio <amadio@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
//
// This file implements a JITEventListener object that tells perf about JITted
// symbols using perf map files (/tmp/perf-%d.map, where %d = pid of process).
//
// Documentation for this perf jit interface is available at:
// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/tools/perf/Documentation/jit-interface.txt
//
//------------------------------------------------------------------------------

#ifdef __linux__

#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Support/ManagedStatic.h"

#include <cstdint>
#include <cstdio>
#include <mutex>

#include <unistd.h>

using namespace llvm;
using namespace llvm::object;

namespace {

  class PerfJITEventListener : public JITEventListener {
  public:
    PerfJITEventListener();
    ~PerfJITEventListener() {
      if (m_Perfmap)
        fclose(m_Perfmap);
    }

    void notifyObjectLoaded(ObjectKey K, const ObjectFile& Obj,
                            const RuntimeDyld::LoadedObjectInfo& L) override;
    void notifyFreeingObject(ObjectKey K) override;

  private:
    std::mutex m_Mutex;
    FILE* m_Perfmap;
  };

  PerfJITEventListener::PerfJITEventListener() {
    char filename[64];
    snprintf(filename, 64, "/tmp/perf-%d.map", getpid());
    m_Perfmap = fopen(filename, "a");
  }

  void PerfJITEventListener::notifyObjectLoaded(
      ObjectKey K, const ObjectFile& Obj,
      const RuntimeDyld::LoadedObjectInfo& L) {

    if (!m_Perfmap)
      return;

    OwningBinary<ObjectFile> DebugObjOwner = L.getObjectForDebug(Obj);
    const ObjectFile& DebugObj = *DebugObjOwner.getBinary();

    // For each symbol, we want to check its address and size
    // if it's a function and write the information to the perf
    // map file, otherwise we just ignore the symbol and any
    // related errors. This implementation is adapted from LLVM:
    // llvm/src/lib/ExecutionEngine/PerfJITEvents/PerfJITEventListener.cpp

    for (const std::pair<SymbolRef, uint64_t>& P :
         computeSymbolSizes(DebugObj)) {
      SymbolRef Sym = P.first;

      Expected<SymbolRef::Type> SymTypeOrErr = Sym.getType();
      if (!SymTypeOrErr) {
        consumeError(SymTypeOrErr.takeError());
        continue;
      }

      SymbolRef::Type SymType = *SymTypeOrErr;
      if (SymType != SymbolRef::ST_Function)
        continue;

      Expected<StringRef> Name = Sym.getName();
      if (!Name) {
        consumeError(Name.takeError());
        continue;
      }

      Expected<uint64_t> AddrOrErr = Sym.getAddress();
      if (!AddrOrErr) {
        consumeError(AddrOrErr.takeError());
        continue;
      }

      uint64_t address = *AddrOrErr;
      uint64_t size = P.second;

      if (size == 0)
        continue;

      std::lock_guard<std::mutex> lock(m_Mutex);
      fprintf(m_Perfmap, "%" PRIx64 " %" PRIx64 " %s\n", address, size, Name->data());
    }

    fflush(m_Perfmap);
  }

  void PerfJITEventListener::notifyFreeingObject(ObjectKey K) {
    // nothing to be done
  }

  llvm::ManagedStatic<PerfJITEventListener> PerfListener;

} // end anonymous namespace

namespace cling {

  JITEventListener* createPerfJITEventListener() { return &*PerfListener; }

} // namespace cling

#endif
