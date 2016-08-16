//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Paths.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace cling {
namespace utils {

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
  llvm::errs() << "  ignoring nonexistent directory \"" << Path << "\"\n";
}

bool SplitPaths(llvm::StringRef PathStr,
                llvm::SmallVectorImpl<llvm::StringRef>& Paths,
                SplitMode Mode, llvm::StringRef Delim, bool Verbose) {
  bool AllExisted = true;
  for (std::pair<llvm::StringRef, llvm::StringRef> Split = PathStr.split(Delim);
       !Split.second.empty(); Split = PathStr.split(Delim)) {
    const bool Exists = llvm::sys::fs::is_directory(Split.first);
    AllExisted = AllExisted && Exists;

    if (!Exists) {
      if (Mode == kFailNonExistant) {
        if (Verbose) {
          // Exiting early, but still log all non-existant paths that we have
          LogNonExistantDirectory(Split.first);
          while (!Split.second.empty()) {
            Split = PathStr.split(Delim);
            if (llvm::sys::fs::is_directory(Split.first)) {
              llvm::errs() << "  ignoring directory that exists \""
                           << Split.first << "\"\n";
            } else
              LogNonExistantDirectory(Split.first);
            Split = Split.second.split(Delim);
          }
          if (!llvm::sys::fs::is_directory(Split.first))
            LogNonExistantDirectory(Split.first);
        }
        return false;
      } else if (Verbose)
        LogNonExistantDirectory(Split.first);
    }

    Paths.push_back(Split.first);
    PathStr = Split.second;
  }

  // Add remaining part
  if (!llvm::sys::fs::is_directory(PathStr)) {
    AllExisted = false;
    if (Mode == kAllowNonExistant)
      Paths.push_back(PathStr);
  } else
    Paths.push_back(PathStr);

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
    llvm::errs() << "Added include paths:\n";
    for (llvm::StringRef Path : PathsChecked)
      llvm::errs() << "  " << Path << "\n";
  }
}
  
} // namespace utils
} // namespace cling
