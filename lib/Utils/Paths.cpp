//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Paths.h"
#include "cling/Utils/Output.h"
#include "clang/Basic/FileManager.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace cling {
namespace utils {

namespace platform {
#if defined(LLVM_ON_UNIX)
  const char* const kEnvDelim = ":";
#elif defined(LLVM_ON_WIN32)
  const char* const kEnvDelim = ";";
#else
  #error "Unknown platform (environmental delimiter)"
#endif
} // namespace platform

bool ExpandEnvVars(std::string& Str, bool Path) {
  std::size_t DPos = Str.find("$");
  while (DPos != std::string::npos) {
    std::size_t SPos = Str.find("/", DPos + 1);
    std::size_t Length = Str.length();

    if (SPos != std::string::npos) // if we found a "/"
      Length = SPos - DPos;

    std::string EnvVar = Str.substr(DPos + 1, Length -1); //"HOME"
    std::string FullPath;
    if (const char* Tok = ::getenv(EnvVar.c_str()))
      FullPath = Tok;

    Str.replace(DPos, Length, FullPath);
    DPos = Str.find("$", DPos + 1); //search for next env variable
  }
  if (!Path)
    return true;
  return llvm::sys::fs::exists(Str.c_str());
}

using namespace clang;

// Adapted from clang/lib/Frontend/CompilerInvocation.cpp

void CopyIncludePaths(const clang::HeaderSearchOptions& Opts,
                      llvm::SmallVectorImpl<std::string>& incpaths,
                      bool withSystem, bool withFlags) {
  if (withFlags && Opts.Sysroot != "/") {
    incpaths.push_back("-isysroot");
    incpaths.push_back(Opts.Sysroot);
  }

  /// User specified include entries.
  for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
    const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
    if (E.IsFramework && E.Group != frontend::Angled)
      llvm::report_fatal_error("Invalid option set!");
    switch (E.Group) {
    case frontend::After:
      if (withFlags) incpaths.push_back("-idirafter");
      break;

    case frontend::Quoted:
      if (withFlags) incpaths.push_back("-iquote");
      break;

    case frontend::System:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-isystem");
      break;

    case frontend::IndexHeaderMap:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-index-header-map");
      if (withFlags) incpaths.push_back(E.IsFramework? "-F" : "-I");
      break;

    case frontend::CSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-c-isystem");
      break;

    case frontend::ExternCSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-extern-c-isystem");
      break;

    case frontend::CXXSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-cxx-isystem");
      break;

    case frontend::ObjCSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-objc-isystem");
      break;

    case frontend::ObjCXXSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-objcxx-isystem");
      break;

    case frontend::Angled:
      if (withFlags) incpaths.push_back(E.IsFramework ? "-F" : "-I");
      break;
    }
    incpaths.push_back(E.Path);
  }

  if (withSystem && !Opts.ResourceDir.empty()) {
    if (withFlags) incpaths.push_back("-resource-dir");
    incpaths.push_back(Opts.ResourceDir);
  }
  if (withSystem && withFlags && !Opts.ModuleCachePath.empty()) {
    incpaths.push_back("-fmodule-cache-path");
    incpaths.push_back(Opts.ModuleCachePath);
  }
  if (withSystem && withFlags && !Opts.UseStandardSystemIncludes)
    incpaths.push_back("-nostdinc");
  if (withSystem && withFlags && !Opts.UseStandardCXXIncludes)
    incpaths.push_back("-nostdinc++");
  if (withSystem && withFlags && Opts.UseLibcxx)
    incpaths.push_back("-stdlib=libc++");
  if (withSystem && withFlags && Opts.Verbose)
    incpaths.push_back("-v");
}

void DumpIncludePaths(const clang::HeaderSearchOptions& Opts,
                      llvm::raw_ostream& Out,
                      bool WithSystem, bool WithFlags) {
  llvm::SmallVector<std::string, 100> IncPaths;
  CopyIncludePaths(Opts, IncPaths, WithSystem, WithFlags);
  // print'em all
  for (unsigned i = 0; i < IncPaths.size(); ++i) {
    Out << IncPaths[i] <<"\n";
  }
}

void LogNonExistantDirectory(llvm::StringRef Path) {
  cling::log() << "  ignoring nonexistent directory \"" << Path << "\"\n";
}

static void LogFileStatus(const char* Prefix, const char* FileType,
                          llvm::StringRef Path) {
  cling::log() << Prefix << " " << FileType << " '" << Path << "'\n";
}

bool LookForFile(const std::vector<const char*>& Args, std::string& Path,
                 const clang::FileManager* FM, const char* FileType) {
  if (llvm::sys::fs::is_regular_file(Path)) {
    if (FileType)
      LogFileStatus("Using", FileType, Path);
    return true;
  }
  if (FileType)
    LogFileStatus("Ignoring", FileType, Path);

  SmallString<1024> FilePath;
  if (FM) {
    FilePath.assign(Path);
    if (FM->FixupRelativePath(FilePath) &&
        llvm::sys::fs::is_regular_file(FilePath)) {
      if (FileType)
        LogFileStatus("Using", FileType, FilePath.str());
      Path = FilePath.str();
      return true;
    }
    // Don't write same same log entry twice when FilePath == Path
    if (FileType && !FilePath.str().equals(Path))
      LogFileStatus("Ignoring", FileType, FilePath);
  }
  else if (llvm::sys::path::is_absolute(Path))
    return false;

  for (std::vector<const char*>::const_iterator It = Args.begin(),
       End = Args.end(); It < End; ++It) {
    const char* Arg = *It;
    // TODO: Suppport '-iquote' and MSVC equivalent
    if (!::strncmp("-I", Arg, 2) || !::strncmp("/I", Arg, 2)) {
      if (!Arg[2]) {
        if (++It >= End)
          break;
        FilePath.assign(*It);
      }
      else
        FilePath.assign(Arg + 2);

      llvm::sys::path::append(FilePath, Path.c_str());
      if (llvm::sys::fs::is_regular_file(FilePath)) {
        if (FileType)
          LogFileStatus("Using", FileType, FilePath.str());
        Path = FilePath.str();
        return true;
      }
      if (FileType)
        LogFileStatus("Ignoring", FileType, FilePath);
    }
  }
  return false;
}

bool SplitPaths(llvm::StringRef PathStr,
                llvm::SmallVectorImpl<llvm::StringRef>& Paths,
                SplitMode Mode, llvm::StringRef Delim, bool Verbose) {
  assert(Delim.size() && "Splitting without a delimiter");

#if defined(LLVM_ON_WIN32)
  // Support using a ':' delimiter on Windows.
  const bool WindowsColon = Delim.equals(":");
#endif

  bool AllExisted = true;
  for (std::pair<llvm::StringRef, llvm::StringRef> Split = PathStr.split(Delim);
       !Split.second.empty(); Split = PathStr.split(Delim)) {

    if (!Split.first.empty()) {
      bool Exists = llvm::sys::fs::is_directory(Split.first);

#if defined(LLVM_ON_WIN32)
    // Because drive letters will have a colon we have to make sure the split
    // occurs at a colon not followed by a path separator.
    if (!Exists && WindowsColon && Split.first.size()==1) {
      // Both clang and cl.exe support '\' and '/' path separators.
      if (Split.second.front() == '\\' || Split.second.front() == '/') {
          const std::pair<llvm::StringRef, llvm::StringRef> Tmp =
              Split.second.split(Delim);
          // Split.first = 'C', but we want 'C:', so Tmp.first.size()+2
          Split.first =
              llvm::StringRef(Split.first.data(), Tmp.first.size() + 2);
          Split.second = Tmp.second;
          Exists = llvm::sys::fs::is_directory(Split.first);
      }
    }
#endif

      AllExisted = AllExisted && Exists;

      if (!Exists) {
        if (Mode == kFailNonExistant) {
          if (Verbose) {
            // Exiting early, but still log all non-existant paths that we have
            LogNonExistantDirectory(Split.first);
            while (!Split.second.empty()) {
              Split = PathStr.split(Delim);
              if (llvm::sys::fs::is_directory(Split.first)) {
                cling::log() << "  ignoring directory that exists \""
                             << Split.first << "\"\n";
              } else
                LogNonExistantDirectory(Split.first);
              Split = Split.second.split(Delim);
            }
            if (!llvm::sys::fs::is_directory(Split.first))
              LogNonExistantDirectory(Split.first);
          }
          return false;
        } else if (Mode == kAllowNonExistant)
          Paths.push_back(Split.first);
        else if (Verbose)
          LogNonExistantDirectory(Split.first);
      } else
        Paths.push_back(Split.first);
    }

    PathStr = Split.second;
  }

  // Trim trailing sep in case of A:B:C:D:
  if (!PathStr.empty() && PathStr.endswith(Delim))
    PathStr = PathStr.substr(0, PathStr.size()-Delim.size());

  if (!PathStr.empty()) {
    if (!llvm::sys::fs::is_directory(PathStr)) {
      AllExisted = false;
      if (Mode == kAllowNonExistant)
        Paths.push_back(PathStr);
      else if (Verbose)
        LogNonExistantDirectory(PathStr);
    } else
      Paths.push_back(PathStr);
  }

  return AllExisted;
}

void AddIncludePaths(llvm::StringRef PathStr, clang::HeaderSearchOptions& HOpts,
                     const char* Delim) {

  llvm::SmallVector<llvm::StringRef, 10> Paths;
  if (Delim && *Delim)
    SplitPaths(PathStr, Paths, kAllowNonExistant, Delim, HOpts.Verbose);
  else
    Paths.push_back(PathStr);

  // Avoid duplicates
  llvm::SmallVector<llvm::StringRef, 10> PathsChecked;
  for (llvm::StringRef Path : Paths) {
    bool Exists = false;
    for (const clang::HeaderSearchOptions::Entry& E : HOpts.UserEntries) {
      if ((Exists = E.Path == Path))
        break;
    }
    if (!Exists)
      PathsChecked.push_back(Path);
  }

  const bool IsFramework = false;
  const bool IsSysRootRelative = true;
  for (llvm::StringRef Path : PathsChecked)
      HOpts.AddPath(Path, clang::frontend::Angled,
                    IsFramework, IsSysRootRelative);

  if (HOpts.Verbose) {
    cling::log() << "Added include paths:\n";
    for (llvm::StringRef Path : PathsChecked)
      cling::log() << "  " << Path << "\n";
  }
}
  
} // namespace utils
} // namespace cling
