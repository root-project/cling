//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_PATHS_H
#define CLING_UTILS_PATHS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class HeaderSearchOptions;
  class FileManager;
}

namespace cling {
  namespace utils {

    namespace platform {
      ///\brief Platform specific delimiter for splitting environment variables.
      /// ':' on Unix, and ';' on Windows
      extern const char* const kEnvDelim;
    }

    ///\brief Replace all $TOKENS in a string with environent variable values.
    ///
    /// \param [in,out] Str - String with tokens to replace (in place)
    /// \param [in] Path - Check if the result is a valid filesystem path.
    ///
    /// \returns When Path is true, return whether Str was expanded to an
    /// existing file-system object.
    /// Return value has no meaning when Path is false.
    ///
    bool ExpandEnvVars(std::string& Str, bool Path = false);

    enum SplitMode {
      kPruneNonExistant,  ///< Don't add non-existant paths into output
      kFailNonExistant,   ///< Fail on any non-existant paths
      kAllowNonExistant   ///< Add all paths whether they exist or not
    };

    ///\brief Collect the constituant paths from a PATH string.
    /// /bin:/usr/bin:/usr/local/bin -> {/bin, /usr/bin, /usr/local/bin}
    ///
    /// All paths returned existed at the time of the call
    /// \param [in] PathStr - The PATH string to be split
    /// \param [out] Paths - All the paths in the string that exist
    /// \param [in] Mode - If any path doesn't exist stop and return false
    /// \param [in] Delim - The delimeter to use
    /// \param [in] Verbose - Whether to print out details as 'clang -v' would
    ///
    /// \return true if all paths existed, otherwise false
    ///
    bool SplitPaths(llvm::StringRef PathStr,
                    llvm::SmallVectorImpl<llvm::StringRef>& Paths,
                    SplitMode Mode = kPruneNonExistant,
                    llvm::StringRef Delim = platform::kEnvDelim,
                    bool Verbose = false);


    ///\brief Look for given file that can be reachable from current working
    /// directory or any user supplied include paths in Args. This is useful
    /// to look for a file (precompiled header) before a Preprocessor instance
    /// has been created.
    ///
    /// \param [in] Args - The argv vector to look for '-I' & '/I' flags
    /// \param [in,out] File - File to look for, may mutate to an absolute path
    /// \param [in] FM - File manger to resolve current dir with (can be null)
    /// \param [in] FileType - File type for logging or nullptr for no logging
    ///
    /// \return true if File is reachable and is a regular file
    ///
    bool LookForFile(const std::vector<const char*>& Args, std::string& File,
                     const clang::FileManager* FM = nullptr,
                     const char* FileType = nullptr);
    
    ///\brief Adds multiple include paths separated by a delimter into the
    /// given HeaderSearchOptions.  This adds the paths but does no further
    /// processing. See Interpreter::AddIncludePaths or CIFactory::createCI
    /// for examples of what needs to be done once the paths have been added.
    ///
    ///\param[in] PathStr - Path(s)
    ///\param[in] Opts - HeaderSearchOptions to add paths into
    ///\param[in] Delim - Delimiter to separate paths or NULL if a single path
    ///
    void AddIncludePaths(llvm::StringRef PathStr,
                         clang::HeaderSearchOptions& Opts,
                         const char* Delim = platform::kEnvDelim);

    ///\brief Write to cling::errs that directory does not exist in a format
    /// matching what 'clang -v' would do
    ///
    void LogNonExistantDirectory(llvm::StringRef Path);

    ///\brief Copies the current include paths into the HeaderSearchOptions.
    ///
    ///\param[in] Opts - HeaderSearchOptions to read from
    ///\param[out] Paths - Vector to output elements into
    ///\param[in] WithSystem - if true, incpaths will also contain system
    ///       include paths (framework, STL etc).
    ///\param[in] WithFlags - if true, each element in incpaths will be prefixed
    ///       with a "-I" or similar, and some entries of incpaths will signal
    ///       a new include path region (e.g. "-cxx-isystem"). Also, flags
    ///       defining header search behavior will be included in incpaths, e.g.
    ///       "-nostdinc".
    ///
    void CopyIncludePaths(const clang::HeaderSearchOptions& Opts,
                          llvm::SmallVectorImpl<std::string>& Paths,
                          bool WithSystem, bool WithFlags);

    ///\brief Prints the current include paths into the HeaderSearchOptions.
    ///
    ///\param[in] Opts - HeaderSearchOptions to read from
    ///\param[in] Out - Stream to dump to
    ///\param[in] WithSystem - dump contain system paths (framework, STL etc).
    ///\param[in] WithFlags - if true, each line will be prefixed
    ///       with a "-I" or similar, and some entries of incpaths will signal
    ///       a new include path region (e.g. "-cxx-isystem"). Also, flags
    ///       defining header search behavior will be included in incpaths, e.g.
    ///       "-nostdinc".
    ///
    void DumpIncludePaths(const clang::HeaderSearchOptions& Opts,
                          llvm::raw_ostream& Out,
                          bool WithSystem, bool WithFlags);
  }
}

#endif // CLING_UTILS_PATHS_H
