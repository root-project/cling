//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_PLATFORM_H
#define CLING_PLATFORM_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace cling {
namespace utils {
namespace platform {

  ///\brief Returns the current working directory
  ///
  std::string GetCwd();

  ///\brief Get the system library paths
  ///
  /// \returns true on success false otherwise
  ///
  bool GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string>& Paths);

#if defined(LLVM_ON_UNIX)

#if defined(__APPLE__)

inline namespace osx {
  ///\brief Get a path to an OSX SDK that can be used for -isysroot. Matches
  ///  1. Version matching the running system
  ///  2. Version that cling was compiled
  ///  3. Highest installed version
  ///
  /// \param [out] SysRoot - The path to the SDK
  /// \param [in] Verbose - Log progress
  ///
  bool GetISysRoot(std::string& SysRoot, bool Verbose = false);

} // namespace osx

#endif // __APPLE__

#elif defined(LLVM_ON_WIN32)

inline namespace windows {

  ///\brief Get an error message from the last Windows API
  ///
  /// \param [in] Prefix - Prefix the message with this (ex. API call name)
  ///
  /// \returns true if ::GetLastError returned an error code
  ///
  bool GetLastErrorAsString(std::string& ErrStr, const char* Prefix = nullptr);

  ///\brief Reports the last Windows API error (currently to cling::errs)
  ///
  /// \param [in] Prefix - Prefix the message with this
  ///
  bool ReportLastError(const char* Prefix = nullptr);

  ///\brief Return true if a given Path is a dynamic library
  ///
  bool IsDLL(const std::string& Path);

  /// \brief Read registry string.
  /// This also supports a means to look for high-versioned keys by use
  /// of a $VERSION placeholder in the key path.
  /// $VERSION in the key path is a placeholder for the version number,
  /// causing the highest value path to be searched for and used.
  /// I.e. "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
  /// There can be additional characters in the component.  Only the numberic
  /// characters are compared.
  ///
  /// \param [in] Key - Key to lookup
  /// \param [in] ValueName - Value to lookup
  /// \param [out] Value - The value of the given key-value pair
  ///
  /// \returns true if key-value existed and was read into Value
  ///
  bool GetSystemRegistryString(const char* Key, const char* ValueName,
                               std::string& Value);

  ///\brief Get a path to an installed VisualStudio directory matching:
  ///  1. Version that cling was compiled
  ///  2. Version that shell is initialized to
  ///  3. Highest installed version
  ///
  /// \param [out] Path - Path to VisualStudio
  /// \param [out] WindSDK - Store the path to the Windows SDK here
  /// \param [out] UniversalSDK - Store the path to the Universal SDK here
  /// \param [in] Verbose - Log progress
  ///
  bool GetVisualStudioDirs(std::string& Path,
                           std::string* WindSDK = nullptr,
                           std::string* UniversalSDK = nullptr,
                           bool Verbose = false);
  
  ///\brief Returns the VisualStudio version cling was compiled with
  int GetVisualStudioVersionCompiledWith();

} // namespace windows
#endif // LLVM_ON_WIN32

} // namespace platform
} // namespace utils
namespace platform = utils::platform;
} // namespace cling

#endif // CLING_PLATFORM_H
