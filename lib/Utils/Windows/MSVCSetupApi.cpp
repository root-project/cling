// <copyright file="Program.cpp" company="Microsoft Corporation">
// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// </copyright>
// <license>
// The MIT License (MIT)
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// </license>

#include "MSVCSetupApi.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

_COM_SMARTPTR_TYPEDEF(ISetupConfiguration, __uuidof(ISetupConfiguration));
_COM_SMARTPTR_TYPEDEF(ISetupConfiguration2, __uuidof(ISetupConfiguration2));
_COM_SMARTPTR_TYPEDEF(ISetupHelper, __uuidof(ISetupHelper));
_COM_SMARTPTR_TYPEDEF(IEnumSetupInstances, __uuidof(IEnumSetupInstances));
_COM_SMARTPTR_TYPEDEF(ISetupInstance, __uuidof(ISetupInstance));
_COM_SMARTPTR_TYPEDEF(ISetupInstance2, __uuidof(ISetupInstance2));

namespace cling {
namespace utils {
namespace platform {
namespace windows {

bool FindVCToolChainViaSetupConfig(std::string& Path, const char* FindVersion) {
#ifndef USE_MSVC_SETUP_API
  return false;
#else
  // FIXME: This really should be done once in the top-level program's main
  // function, as it may have already been initialized with a different
  // threading model otherwise.
  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::SingleThreaded);
  HRESULT HR;

  // _com_ptr_t will throw a _com_error if a COM calls fail.
  // The LLVM coding standards forbid exception handling, so we'll have to
  // stop them from being thrown in the first place.
  // The destructor will put the regular error handler back when we leave
  // this scope.
  struct SuppressCOMErrorsRAII {
    static void __stdcall handler(HRESULT hr, IErrorInfo *perrinfo) {}

    SuppressCOMErrorsRAII() { _set_com_error_handler(handler); }

    ~SuppressCOMErrorsRAII() { _set_com_error_handler(_com_raise_error); }

  } COMErrorSuppressor;

  ISetupConfigurationPtr Query;
  HR = Query.CreateInstance(__uuidof(SetupConfiguration));
  if (FAILED(HR))
    return false;

  IEnumSetupInstancesPtr EnumInstances;
  HR = ISetupConfiguration2Ptr(Query)->EnumAllInstances(&EnumInstances);
  if (FAILED(HR))
    return false;

  ISetupInstancePtr Instance;
  HR = EnumInstances->Next(1, &Instance, nullptr);
  if (HR != S_OK)
    return false;

  uint64_t BestVersion = 0;
  if (FindVersion) {
    bstr_t VersionString(FindVersion);
    HR = ISetupHelperPtr(Query)->ParseVersion(VersionString, &BestVersion);
    if (FAILED(HR))
      return false;
  }

  ISetupInstancePtr NewestInstance;
  do {
    bstr_t VersionString;
    uint64_t VersionNum;
    HR = Instance->GetInstallationVersion(VersionString.GetAddress());
    if (FAILED(HR))
      continue;
    HR = ISetupHelperPtr(Query)->ParseVersion(VersionString, &VersionNum);
    if (FAILED(HR))
      continue;
    if (FindVersion) {
      if (VersionNum == BestVersion) {
        NewestInstance = Instance;
        break;
      }
    } else if (!BestVersion || (VersionNum > BestVersion)) {
      NewestInstance = Instance;
      BestVersion = VersionNum;
    }
  } while ((HR = EnumInstances->Next(1, &Instance, nullptr)) == S_OK);

  if (!NewestInstance)
    return false;

  bstr_t VCPathWide;
  HR = NewestInstance->ResolvePath(L"VC", VCPathWide.GetAddress());
  if (FAILED(HR))
    return false;

  std::string VCRootPath;
  llvm::convertWideToUTF8(std::wstring(VCPathWide), VCRootPath);

  llvm::SmallString<256> ToolsVersionFilePath(VCRootPath);
  llvm::sys::path::append(ToolsVersionFilePath, "Auxiliary", "Build",
                          "Microsoft.VCToolsVersion.default.txt");

  auto ToolsVersionFile = llvm::MemoryBuffer::getFile(ToolsVersionFilePath);
  if (!ToolsVersionFile)
    return false;

  llvm::SmallString<256> ToolchainPath(VCRootPath);
  llvm::sys::path::append(ToolchainPath, "Tools", "MSVC",
                          ToolsVersionFile->get()->getBuffer().rtrim());
  if (!llvm::sys::fs::is_directory(ToolchainPath))
    return false;

  Path = ToolchainPath.str();
  return true;
#endif
}

}
}
}
}
