//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
// author:  Alexander Penev <alexander_penev@yahoo.com>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Utils/Paths.h"
#include "cling/Utils/Platform.h"
#include "cling/Utils/Output.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/WithColor.h"

#include <algorithm>
#include <list>
#include <string>
#include <unordered_set>
#include <vector>


#ifdef LLVM_ON_UNIX
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#endif // LLVM_ON_UNIX

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/stat.h>
#undef LC_LOAD_DYLIB
#undef LC_RPATH
#endif // __APPLE__

#ifdef _WIN32
#include <windows.h>
#include <libloaderapi.h> // For GetModuleFileNameA
#include <memoryapi.h> // For VirtualQuery
#endif

// FIXME: Implement debugging output stream in cling.
constexpr unsigned DEBUG = 0;

namespace {

using BasePath = std::string;

// This is a GNU implementation of hash used in bloom filter!
static uint32_t GNUHash(llvm::StringRef S) {
  uint32_t H = 5381;
  for (uint8_t C : S)
    H = (H << 5) + H + C;
  return H;
}

constexpr uint32_t log2u(std::uint32_t n) {
  return (n > 1) ? 1 + log2u(n >> 1) : 0;
}

struct BloomFilter {

  // https://hur.st/bloomfilter
  //
  // n = ceil(m / (-k / log(1 - exp(log(p) / k))))
  // p = pow(1 - exp(-k / (m / n)), k)
  // m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
  // k = round((m / n) * log(2));
  //
  // n = symbolsCount
  // p = 0.02
  // k = 2 (k1=GNUHash and k2=GNUHash >> bloomShift)
  // m = ceil((symbolsCount * log(p)) / log(1 / pow(2, log(2))));
  // bloomShift = min(5 for bits=32 or 6 for bits=64, log2(symbolsCount))
  // bloomSize = ceil((-1.44 * n * log2f(p)) / bits)

  const int m_Bits = 8 * sizeof(uint64_t);
  const float m_P = 0.02;

  bool m_IsInitialized = false;
  uint32_t m_SymbolsCount = 0;
  uint32_t m_BloomSize = 0;
  uint32_t m_BloomShift = 0;
  std::vector<uint64_t> m_BloomTable;

  bool TestHash(uint32_t hash) const {
    // This function is superhot. No branches here, breaks inlining and makes
    // overall performance around 4x slower.
    assert(m_IsInitialized && "Not yet initialized!");
    uint32_t hash2 = hash >> m_BloomShift;
    uint32_t n = (hash >> log2u(m_Bits)) % m_BloomSize;
    uint64_t mask = ((1ULL << (hash % m_Bits)) | (1ULL << (hash2 % m_Bits)));
    return (mask & m_BloomTable[n]) == mask;
  }

  void AddHash(uint32_t hash) {
    assert(m_IsInitialized && "Not yet initialized!");
    uint32_t hash2 = hash >> m_BloomShift;
    uint32_t n = (hash >> log2u(m_Bits)) % m_BloomSize;
    uint64_t mask = ((1ULL << (hash % m_Bits)) | (1ULL << (hash2 % m_Bits)));
    m_BloomTable[n] |= mask;
  }

  void ResizeTable(uint32_t newSymbolsCount) {
    assert(m_SymbolsCount == 0 && "Not supported yet!");
    m_SymbolsCount = newSymbolsCount;
    m_BloomSize = ceil((-1.44f * m_SymbolsCount * log2f(m_P)) / m_Bits);
    m_BloomShift = std::min(6u, log2u(m_SymbolsCount));
    m_BloomTable.resize(m_BloomSize);
  }

};

/// An efficient representation of a full path to a library which does not
/// duplicate common path patterns reducing the overall memory footprint.
///
/// For example, `/home/.../lib/libA.so`, m_Path will contain a pointer
/// to  `/home/.../lib/`
/// will be stored and .second `libA.so`.
/// This approach reduces the duplicate paths as at one location there may be
/// plenty of libraries.
struct LibraryPath {
  const BasePath& m_Path;
  std::string m_LibName;
  BloomFilter m_Filter;
  llvm::StringSet<> m_Symbols;
  //std::vector<const LibraryPath*> m_LibDeps;

  LibraryPath(const BasePath& Path, const std::string& LibName)
    : m_Path(Path), m_LibName(LibName) {
  }

  bool operator==(const LibraryPath &other) const {
    return (&m_Path == &other.m_Path || m_Path == other.m_Path) &&
      m_LibName == other.m_LibName;
  }

  const std::string GetFullName() const {
    llvm::SmallString<512> Vec(m_Path);
    llvm::sys::path::append(Vec, llvm::StringRef(m_LibName));
    return Vec.str().str();
  }

  void AddBloom(llvm::StringRef symbol) {
    m_Filter.AddHash(GNUHash(symbol));
  }

  llvm::StringRef AddSymbol(const std::string& symbol) {
    auto it = m_Symbols.insert(symbol);
    return it.first->getKey();
  }

  bool hasBloomFilter() const {
    return m_Filter.m_IsInitialized;
  }

  bool isBloomFilterEmpty() const {
    assert(m_Filter.m_IsInitialized && "Bloom filter not initialized!");
    return m_Filter.m_SymbolsCount == 0;
  }

  void InitializeBloomFilter(uint32_t newSymbolsCount) {
    assert(!m_Filter.m_IsInitialized &&
           "Cannot re-initialize non-empty filter!");
    m_Filter.m_IsInitialized = true;
    m_Filter.ResizeTable(newSymbolsCount);
  }

  bool MayExistSymbol(uint32_t hash) const {
    // The library had no symbols and the bloom filter is empty.
    if (isBloomFilterEmpty())
      return false;

    return m_Filter.TestHash(hash);
  }

  bool ExistSymbol(llvm::StringRef symbol) const {
    return m_Symbols.find(symbol) != m_Symbols.end();
  }
};


/// A helper class keeping track of loaded libraries. It implements a fast
/// search O(1) while keeping deterministic iterability in a memory efficient
/// way. The underlying set uses a custom hasher for better efficiency given the
/// specific problem where the library names (m_LibName) are relatively short
/// strings and the base paths (m_Path) are repetitive long strings.
class LibraryPaths {
  struct LibraryPathHashFn {
    size_t operator()(const LibraryPath& item) const {
      return std::hash<size_t>()(item.m_Path.length()) ^
        std::hash<std::string>()(item.m_LibName);
    }
  };

  std::vector<const LibraryPath*> m_Libs;
  std::unordered_set<LibraryPath, LibraryPathHashFn> m_LibsH;
public:
  bool HasRegisteredLib(const LibraryPath& Lib) const {
    return m_LibsH.count(Lib);
  }

  const LibraryPath* GetRegisteredLib(const LibraryPath& Lib) const {
    auto search = m_LibsH.find(Lib);
    if (search != m_LibsH.end())
      return &(*search);
    return nullptr;
  }

  const LibraryPath* RegisterLib(const LibraryPath& Lib) {
    auto it = m_LibsH.insert(Lib);
    assert(it.second && "Already registered!");
    m_Libs.push_back(&*it.first);
    return &*it.first;
  }

  void UnregisterLib(const LibraryPath& Lib) {
    auto found = m_LibsH.find(Lib);
    if (found == m_LibsH.end())
      return;

    m_Libs.erase(std::find(m_Libs.begin(), m_Libs.end(), &*found));
    m_LibsH.erase(found);
  }

  size_t size() const {
    assert(m_Libs.size() == m_LibsH.size());
    return m_Libs.size();
  }

  const std::vector<const LibraryPath*>& GetLibraries() const {
    return m_Libs;
  }
};

#ifndef _WIN32
// Cached version of system function lstat
static inline mode_t cached_lstat(const char *path) {
  static llvm::StringMap<mode_t> lstat_cache;

  // If already cached - retun cached result
  auto it = lstat_cache.find(path);
  if (it != lstat_cache.end())
    return it->second;

  // If result not in cache - call system function and cache result
  struct stat buf;
  mode_t st_mode = (lstat(path, &buf) == -1) ? 0 : buf.st_mode;
  lstat_cache.insert(std::pair<llvm::StringRef, mode_t>(path, st_mode));
  return st_mode;
}

// Cached version of system function readlink
static inline llvm::StringRef cached_readlink(const char* pathname) {
  static llvm::StringMap<std::string> readlink_cache;

  // If already cached - retun cached result
  auto it = readlink_cache.find(pathname);
  if (it != readlink_cache.end())
    return llvm::StringRef(it->second);

  // If result not in cache - call system function and cache result
  char buf[PATH_MAX];
  ssize_t len;
  if ((len = readlink(pathname, buf, sizeof(buf))) != -1) {
    buf[len] = '\0';
    std::string s(buf);
    readlink_cache.insert(std::pair<llvm::StringRef, std::string>(pathname, s));
    return readlink_cache[pathname];
  }
  return "";
}
#endif

// Cached version of system function realpath
std::string cached_realpath(llvm::StringRef path, llvm::StringRef base_path = "",
                            bool is_base_path_real = false,
                            long symlooplevel = 40) {
  if (path.empty()) {
    errno = ENOENT;
    return "";
  }

  if (!symlooplevel) {
    errno = ELOOP;
    return "";
  }

  // If already cached - retun cached result
  static llvm::StringMap<std::pair<std::string,int>> cache;
  bool relative_path = llvm::sys::path::is_relative(path);
  if (!relative_path) {
    auto it = cache.find(path);
    if (it != cache.end()) {
      errno = it->second.second;
      return it->second.first;
    }
  }

  // If result not in cache - call system function and cache result

  llvm::StringRef sep(llvm::sys::path::get_separator());
  llvm::SmallString<256> result(sep);
#ifndef _WIN32
  llvm::SmallVector<llvm::StringRef, 16> p;

  // Relative or absolute path
  if (relative_path) {
    if (is_base_path_real) {
      result.assign(base_path);
    } else {
      if (path[0] == '~' && (path.size() == 1 || llvm::sys::path::is_separator(path[1]))) {
        static llvm::SmallString<128> home;
        if (home.str().empty())
          llvm::sys::path::home_directory(home);
        llvm::StringRef(home).split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      } else if (base_path.empty()) {
        static llvm::SmallString<256> current_path;
        if (current_path.str().empty())
          llvm::sys::fs::current_path(current_path);
        llvm::StringRef(current_path).split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      } else {
        base_path.split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      }
    }
  }
  path.split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);

  // Handle path list items
  for (auto item : p) {
    if (item.startswith(".")) {
      if (item == "..") {
        size_t s = result.rfind(sep);
        if (s != llvm::StringRef::npos) result.resize(s);
        if (result.empty()) result = sep;
      }
      continue;
    } else if (item == "~") {
      continue;
    }

    size_t old_size = result.size();
    llvm::sys::path::append(result, item);
    mode_t st_mode = cached_lstat(result.c_str());
    if (S_ISLNK(st_mode)) {
      llvm::StringRef symlink = cached_readlink(result.c_str());
      if (llvm::sys::path::is_relative(symlink)) {
        result.set_size(old_size);
        result = cached_realpath(symlink, result, true, symlooplevel - 1);
      } else {
        result = cached_realpath(symlink, "", true, symlooplevel - 1);
      }
    } else if (st_mode == 0) {
      cache.insert(std::pair<llvm::StringRef, std::pair<std::string,int>>(
        path,
        std::pair<std::string,int>("",ENOENT))
      );
      errno = ENOENT;
      return "";
    }
  }
#else
  llvm::sys::fs::real_path(path, result);
#endif
  cache.insert(std::pair<llvm::StringRef, std::pair<std::string,int>>(
    path,
    std::pair<std::string,int>(result.str().str(),errno))
  );
  return result.str().str();
}

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
static Expected<StringRef> getDynamicStrTab(const ELFFile<ELFT>* Elf) {
  auto DynamicEntriesOrError = Elf->dynamicEntries();
  if (!DynamicEntriesOrError)
    return DynamicEntriesOrError.takeError();

  for (const typename ELFT::Dyn& Dyn : *DynamicEntriesOrError) {
    if (Dyn.d_tag == ELF::DT_STRTAB) {
      auto MappedAddrOrError = Elf->toMappedAddr(Dyn.getPtr());
      if (!MappedAddrOrError)
        return MappedAddrOrError.takeError();
      return StringRef(reinterpret_cast<const char *>(*MappedAddrOrError));
    }
  }

  // If the dynamic segment is not present, we fall back on the sections.
  auto SectionsOrError = Elf->sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  for (const typename ELFT::Shdr &Sec : *SectionsOrError) {
    if (Sec.sh_type == ELF::SHT_DYNSYM)
      return Elf->getStringTableForSymtab(Sec);
  }

  return createError("dynamic string table not found");
}

static llvm::StringRef GetGnuHashSection(llvm::object::ObjectFile *file) {
  for (auto S : file->sections()) {
    llvm::StringRef name;
    S.getName(name);
    if (name == ".gnu.hash") {
      return llvm::cantFail(S.getContents());
    }
  }
  return "";
}

/// Bloom filter is a stochastic data structure which can tell us if a symbol
/// name does not exist in a library with 100% certainty. If it tells us it
/// exists this may not be true:
/// https://blogs.oracle.com/solaris/gnu-hash-elf-sections-v2
///
/// ELF has this optimization in the new linkers by default, It is stored in the
/// gnu.hash section of the object file.
///
///\returns true if the symbol may be in the library.
static bool MayExistInElfObjectFile(llvm::object::ObjectFile *soFile,
                                    uint32_t hash) {
  assert(soFile->isELF() && "Not ELF");

  // Compute the platform bitness -- either 64 or 32.
  const unsigned bits = 8 * soFile->getBytesInAddress();

  llvm::StringRef contents = GetGnuHashSection(soFile);
  if (contents.size() < 16)
    // We need to search if the library doesn't have .gnu.hash section!
    return true;
  const char* hashContent = contents.data();

  // See https://flapenguin.me/2017/05/10/elf-lookup-dt-gnu-hash/ for .gnu.hash
  // table layout.
  uint32_t maskWords = *reinterpret_cast<const uint32_t *>(hashContent + 8);
  uint32_t shift2 = *reinterpret_cast<const uint32_t *>(hashContent + 12);
  uint32_t hash2 = hash >> shift2;
  uint32_t n = (hash / bits) % maskWords;

  const char *bloomfilter = hashContent + 16;
  const char *hash_pos = bloomfilter + n*(bits/8); // * (Bits / 8)
  uint64_t word = *reinterpret_cast<const uint64_t *>(hash_pos);
  uint64_t bitmask = ( (1ULL << (hash % bits)) | (1ULL << (hash2 % bits)));
  return  (bitmask & word) == bitmask;
}

} // anon namespace

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
std::string GetExecutablePath() {
   // This just needs to be some symbol in the binary; C++ doesn't
   // allow taking the address of ::main however.
   return cling::DynamicLibraryManager::getSymbolLocation(&GetExecutablePath);
}

namespace cling {
  class Dyld {
    struct BasePathHashFunction {
      size_t operator()(const BasePath& item) const {
        return std::hash<std::string>()(item);
      }
    };

    struct BasePathEqFunction {
      size_t operator()(const BasePath& l, const BasePath& r) const {
        return &l == &r || l == r;
      }
    };
    /// A memory efficient llvm::VectorSet. The class provides O(1) search
    /// complexity. It is tuned to compare BasePaths first by checking the
    /// address and then the representation which models the base path reuse.
    class BasePaths {
    public:
      std::unordered_set<BasePath, BasePathHashFunction,
                         BasePathEqFunction> m_Paths;
    public:
      const BasePath& RegisterBasePath(const std::string& Path,
                                       bool* WasInserted = nullptr) {
        auto it = m_Paths.insert(Path);
        if (WasInserted)
          *WasInserted = it.second;

        return *it.first;
      }

      bool Contains(StringRef Path) {
        return m_Paths.count(Path);
      }
    };

    bool m_FirstRun = true;
    bool m_FirstRunSysLib = true;
    bool m_UseBloomFilter = true;
    bool m_UseHashTable = true;

    const cling::DynamicLibraryManager& m_DynamicLibraryManager;

    /// The basename of `/home/.../lib/libA.so`,
    /// m_BasePaths will contain `/home/.../lib/`
    BasePaths m_BasePaths;

    LibraryPaths m_Libraries;
    LibraryPaths m_SysLibraries;
    /// Contains a set of libraries which we gave to the user via ResolveSymbol
    /// call and next time we should check if the user loaded them to avoid
    /// useless iterations.
    LibraryPaths m_QueriedLibraries;

    using PermanentlyIgnoreCallbackProto = std::function<bool(llvm::StringRef)>;
    const PermanentlyIgnoreCallbackProto m_ShouldPermanentlyIgnoreCallback;
    const llvm::StringRef m_ExecutableFormat;

    /// Scan for shared objects which are not yet loaded. They are a our symbol
    /// resolution candidate sources.
    /// NOTE: We only scan not loaded shared objects.
    /// \param[in] searchSystemLibraries - whether to decent to standard system
    ///            locations for shared objects.
    void ScanForLibraries(bool searchSystemLibraries = false);

    /// Builds a bloom filter lookup optimization.
    void BuildBloomFilter(LibraryPath* Lib, llvm::object::ObjectFile *BinObjFile,
                          unsigned IgnoreSymbolFlags = 0) const;


    /// Looks up symbols from a an object file, representing the library.
    ///\param[in] Lib - full path to the library.
    ///\param[in] mangledName - the mangled name to look for.
    ///\param[in] IgnoreSymbolFlags - The symbols to ignore upon a match.
    ///\returns true on success.
    bool ContainsSymbol(const LibraryPath* Lib, StringRef mangledName,
                        unsigned IgnoreSymbolFlags = 0) const;

    bool ShouldPermanentlyIgnore(StringRef FileName) const;
    void dumpDebugInfo() const;
  public:
    Dyld(const cling::DynamicLibraryManager &DLM,
         PermanentlyIgnoreCallbackProto shouldIgnore,
         llvm::StringRef execFormat)
      : m_DynamicLibraryManager(DLM),
        m_ShouldPermanentlyIgnoreCallback(shouldIgnore),
        m_ExecutableFormat(execFormat) { }

    ~Dyld(){};

    std::string searchLibrariesForSymbol(StringRef mangledName,
                                         bool searchSystem);
  };

  std::string RPathToStr(llvm::SmallVector<llvm::StringRef,2> V) {
    std::string result;
    for (auto item : V)
      result += item.str() + ",";
    if (!result.empty())
      result.pop_back();
    return result;
  }

  void CombinePaths(std::string& P1, const char* P2) {
    if (!P2 || !P2[0]) return;
    if (!P1.empty())
      P1 += llvm::sys::EnvPathSeparator;
    P1 += P2;
  }

  template <class ELFT>
  void HandleDynTab(const ELFFile<ELFT>* Elf, llvm::StringRef FileName,
                    llvm::SmallVector<llvm::StringRef,2>& RPath,
                    llvm::SmallVector<llvm::StringRef,2>& RunPath,
                    std::vector<StringRef>& Deps,
                    bool& isPIEExecutable) {
    const char *Data = "";
    if (Expected<StringRef> StrTabOrErr = getDynamicStrTab(Elf))
      Data = StrTabOrErr.get().data();

    isPIEExecutable = false;

    auto DynamicEntriesOrError = Elf->dynamicEntries();
    if (!DynamicEntriesOrError) {
       cling::errs() << "Dyld: failed to read dynamic entries in"
                     << "'" << FileName.str() << "'\n";
       return;
    }

    for (const typename ELFT::Dyn& Dyn : *DynamicEntriesOrError) {
      switch (Dyn.d_tag) {
        case ELF::DT_NEEDED:
          Deps.push_back(Data + Dyn.d_un.d_val);
          break;
        case ELF::DT_RPATH:
          SplitPaths(Data + Dyn.d_un.d_val, RPath, utils::kAllowNonExistant, platform::kEnvDelim, false);
          break;
        case ELF::DT_RUNPATH:
          SplitPaths(Data + Dyn.d_un.d_val, RunPath, utils::kAllowNonExistant, platform::kEnvDelim, false);
          break;
        case ELF::DT_FLAGS_1:
          // Check if this is not a pie executable.
          if (Dyn.d_un.d_val & llvm::ELF::DF_1_PIE)
            isPIEExecutable = true;
          break;
        // (Dyn.d_tag == ELF::DT_NULL) continue;
        // (Dyn.d_tag == ELF::DT_AUXILIARY || Dyn.d_tag == ELF::DT_FILTER)
      }
    }
  }

  void Dyld::ScanForLibraries(bool searchSystemLibraries/* = false*/) {

    const auto &searchPaths = m_DynamicLibraryManager.getSearchPaths();

    if (DEBUG > 7) {
      cling::errs() << "Dyld::ScanForLibraries: system=" << (searchSystemLibraries?"true":"false") << "\n";
      for (const DynamicLibraryManager::SearchPathInfo &Info : searchPaths)
        cling::errs() << ">>>" << Info.Path << ", " << (Info.IsUser?"user\n":"system\n");
    }

    llvm::SmallSet<const BasePath*, 32> ScannedPaths;

    for (const DynamicLibraryManager::SearchPathInfo &Info : searchPaths) {
      if (Info.IsUser != searchSystemLibraries) {
        // Examples which we should handle.
        // File                      Real
        // /lib/1/1.so               /lib/1/1.so  // file
        // /lib/1/2.so->/lib/1/1.so  /lib/1/1.so  // file local link
        // /lib/1/3.so->/lib/3/1.so  /lib/3/1.so  // file external link
        // /lib/2->/lib/1                         // path link
        // /lib/2/1.so               /lib/1/1.so  // path link, file
        // /lib/2/2.so->/lib/1/1.so  /lib/1/1.so  // path link, file local link
        // /lib/2/3.so->/lib/3/1.so  /lib/3/1.so  // path link, file external link
        //
        // /lib/3/1.so
        // /lib/3/2.so->/system/lib/s.so
        // /lib/3/3.so
        // /system/lib/1.so
        //
        // libL.so NEEDED/RPATH libR.so    /lib/some-rpath/libR.so  // needed/dependedt library in libL.so RPATH/RUNPATH or other (in)direct dep
        //
        // Paths = /lib/1 : /lib/2 : /lib/3

        // m_BasePaths = ["/lib/1", "/lib/3", "/system/lib"]
        // m_*Libraries  = [<0,"1.so">, <1,"1.so">, <2,"s.so">, <1,"3.so">]

        if (DEBUG > 7) {
          cling::errs() << "Dyld::ScanForLibraries Iter:" << Info.Path << " -> ";
        }
        std::string RealPath = cached_realpath(Info.Path);

        llvm::StringRef DirPath(RealPath);
        if (DEBUG > 7) {
          cling::errs() << RealPath << "\n";
        }

        if (!llvm::sys::fs::is_directory(DirPath) || DirPath.empty())
          continue;

        // Already searched?
        const BasePath &ScannedBPath = m_BasePaths.RegisterBasePath(RealPath);
        if (ScannedPaths.count(&ScannedBPath)) {
          if (DEBUG > 7) {
            cling::errs() << "Dyld::ScanForLibraries Already scanned: " << RealPath << "\n";
          }
          continue;
        }

        // FileName must be always full/absolute/resolved file name.
        std::function<void(llvm::StringRef, unsigned)> HandleLib =
          [&](llvm::StringRef FileName, unsigned level) {

          if (DEBUG > 7) {
            cling::errs() << "Dyld::ScanForLibraries HandleLib:" << FileName.str()
               << ", level=" << level << " -> ";
          }

          llvm::StringRef FileRealPath = llvm::sys::path::parent_path(FileName);
          llvm::StringRef FileRealName = llvm::sys::path::filename(FileName);
          const BasePath& BaseP = m_BasePaths.RegisterBasePath(FileRealPath.str());
          LibraryPath LibPath(BaseP, FileRealName); //bp, str

          if (m_SysLibraries.GetRegisteredLib(LibPath) ||
              m_Libraries.GetRegisteredLib(LibPath)) {
            if (DEBUG > 7) {
              cling::errs() << "Already handled!!!\n";
            }
            return;
          }

          if (ShouldPermanentlyIgnore(FileName)) {
            if (DEBUG > 7) {
              cling::errs() << "PermanentlyIgnored!!!\n";
            }
            return;
          }

          if (searchSystemLibraries)
            m_SysLibraries.RegisterLib(LibPath);
          else
            m_Libraries.RegisterLib(LibPath);

          // Handle lib dependencies
          llvm::SmallVector<llvm::StringRef, 2> RPath;
          llvm::SmallVector<llvm::StringRef, 2> RunPath;
          std::vector<StringRef> Deps;
          auto ObjFileOrErr =
            llvm::object::ObjectFile::createObjectFile(FileName);
          if (llvm::Error Err = ObjFileOrErr.takeError()) {
            if (DEBUG > 1) {
              std::string Message;
              handleAllErrors(std::move(Err), [&](llvm::ErrorInfoBase &EIB) {
                Message += EIB.message() + "; ";
              });
              cling::errs()
                << "Dyld::ScanForLibraries: Failed to read object file "
                << FileName.str() << " Errors: " << Message << "\n";
            }
            return;
          }
          llvm::object::ObjectFile *BinObjF = ObjFileOrErr.get().getBinary();
          if (BinObjF->isELF()) {
            bool isPIEExecutable = false;

            if (const auto* ELF = dyn_cast<ELF32LEObjectFile>(BinObjF))
              HandleDynTab(ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                           isPIEExecutable);
            else if (const auto* ELF = dyn_cast<ELF32BEObjectFile>(BinObjF))
              HandleDynTab(ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                           isPIEExecutable);
            else if (const auto* ELF = dyn_cast<ELF64LEObjectFile>(BinObjF))
              HandleDynTab(ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                           isPIEExecutable);
            else if (const auto* ELF = dyn_cast<ELF64BEObjectFile>(BinObjF))
              HandleDynTab(ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                           isPIEExecutable);

            if ((level == 0) && isPIEExecutable) {
              if (searchSystemLibraries)
                m_SysLibraries.UnregisterLib(LibPath);
              else
                m_Libraries.UnregisterLib(LibPath);
              return;
            }
          } else if (BinObjF->isMachO()) {
            MachOObjectFile *Obj = (MachOObjectFile*)BinObjF;
            for (const auto &Command : Obj->load_commands()) {
              if (Command.C.cmd == MachO::LC_LOAD_DYLIB) {
                  //Command.C.cmd == MachO::LC_ID_DYLIB ||
                  //Command.C.cmd == MachO::LC_LOAD_WEAK_DYLIB ||
                  //Command.C.cmd == MachO::LC_REEXPORT_DYLIB ||
                  //Command.C.cmd == MachO::LC_LAZY_LOAD_DYLIB ||
                  //Command.C.cmd == MachO::LC_LOAD_UPWARD_DYLIB ||
                MachO::dylib_command dylibCmd =
                  Obj->getDylibIDLoadCommand(Command);
                Deps.push_back(StringRef(Command.Ptr + dylibCmd.dylib.name));
              }
              else if (Command.C.cmd == MachO::LC_RPATH) {
                MachO::rpath_command rpathCmd = Obj->getRpathCommand(Command);
                SplitPaths(Command.Ptr + rpathCmd.path, RPath, utils::kAllowNonExistant, platform::kEnvDelim, false);
              }
            }
          } else if (BinObjF->isCOFF()) {
            // TODO: COFF support
          }

          if (DEBUG > 7) {
            cling::errs() << "Dyld::ScanForLibraries: Deps Info:\n";
            cling::errs() << "Dyld::ScanForLibraries:   RPATH=" << RPathToStr(RPath) << "\n";
            cling::errs() << "Dyld::ScanForLibraries:   RUNPATH=" << RPathToStr(RunPath) << "\n";
            int x = 0;
            for (StringRef dep : Deps)
              cling::errs() << "Dyld::ScanForLibraries:   Deps[" << x++ << "]=" << dep.str() << "\n";
          }

          // Heuristics for workaround performance problems:
          // (H1) If RPATH and RUNPATH == "" -> skip handling Deps
          if (RPath.empty() && RunPath.empty()) {
            if (DEBUG > 7) {
              cling::errs() << "Dyld::ScanForLibraries: Skip all deps by Heuristic1: " << FileName.str() << "\n";
            }
            return;
          };
          // (H2) If RPATH subset of LD_LIBRARY_PATH &&
          //         RUNPATH subset of LD_LIBRARY_PATH  -> skip handling Deps
          if (std::all_of(RPath.begin(), RPath.end(), [&](StringRef item){ return std::any_of(searchPaths.begin(), searchPaths.end(), [&](DynamicLibraryManager::SearchPathInfo item1){ return item==item1.Path; }); }) &&
              std::all_of(RunPath.begin(), RunPath.end(), [&](StringRef item){ return std::any_of(searchPaths.begin(), searchPaths.end(), [&](DynamicLibraryManager::SearchPathInfo item1){ return item==item1.Path; }); }) ) {
            if (DEBUG > 7) {
              cling::errs() << "Dyld::ScanForLibraries: Skip all deps by Heuristic2: " << FileName.str() << "\n";
            }
            return;
          }

          // Handle dependencies
          for (StringRef dep : Deps) {
            std::string dep_full =
              m_DynamicLibraryManager.lookupLibrary(dep, RPath, RunPath, FileName, false);
              HandleLib(dep_full, level + 1);
          }

        };

        if (DEBUG > 7) {
          cling::errs() << "Dyld::ScanForLibraries: Iterator: " << DirPath << "\n";
        }
        std::error_code EC;
        for (llvm::sys::fs::directory_iterator DirIt(DirPath, EC), DirEnd;
             DirIt != DirEnd && !EC; DirIt.increment(EC)) {

          if (DEBUG > 7) {
            cling::errs() << "Dyld::ScanForLibraries: Iterator >>> " <<
              DirIt->path() << ", type=" << (short)(DirIt->type()) << "\n";
          }

          const llvm::sys::fs::file_type ft = DirIt->type();
          if (ft == llvm::sys::fs::file_type::regular_file) {
              HandleLib(DirIt->path(), 0);
          } else if (ft == llvm::sys::fs::file_type::symlink_file) {
              std::string DepFileName_str = cached_realpath(DirIt->path());
              llvm::StringRef DepFileName = DepFileName_str;
              assert(!llvm::sys::fs::is_symlink_file(DepFileName));
              if (!llvm::sys::fs::is_directory(DepFileName))
                HandleLib(DepFileName, 0);
          }
        }

        // Register the DirPath as fully scanned.
        ScannedPaths.insert(&ScannedBPath);
      }
    }
  }

  void Dyld::BuildBloomFilter(LibraryPath* Lib,
                              llvm::object::ObjectFile *BinObjFile,
                              unsigned IgnoreSymbolFlags /*= 0*/) const {
    assert(m_UseBloomFilter && "Bloom filter is disabled");
    assert(!Lib->hasBloomFilter() && "Already built!");

    using namespace llvm;
    using namespace llvm::object;

    if (DEBUG > 7) {
      cling::errs()<< "Dyld::BuildBloomFilter: Start building Bloom filter for: "
        << Lib->GetFullName() << "\n";
    }

    // If BloomFilter is empty then build it.
    // Count Symbols and generate BloomFilter
    uint32_t SymbolsCount = 0;
    std::list<llvm::StringRef> symbols;
    for (const llvm::object::SymbolRef &S : BinObjFile->symbols()) {
      uint32_t Flags = S.getFlags();
      // Do not insert in the table symbols flagged to ignore.
      if (Flags & IgnoreSymbolFlags)
        continue;

      // Note, we are at last resort and loading library based on a weak
      // symbol is allowed. Otherwise, the JIT will issue an unresolved
      // symbol error.
      //
      // There are other weak symbol kinds (marked as 'V') to denote
      // typeinfo and vtables. It is unclear whether we should load such
      // libraries or from which library we should resolve the symbol.
      // We seem to not have a way to differentiate it from the symbol API.

      llvm::Expected<llvm::StringRef> SymNameErr = S.getName();
      if (!SymNameErr) {
        cling::errs()<< "Dyld::BuildBloomFilter: Failed to read symbol "
                     << SymNameErr.get() << "\n";
        continue;
      }

      if (SymNameErr.get().empty())
        continue;

      ++SymbolsCount;
      symbols.push_back(SymNameErr.get());
    }

    if (BinObjFile->isELF()) {
      // ELF file format has .dynstr section for the dynamic symbol table.
      const auto *ElfObj = cast<llvm::object::ELFObjectFileBase>(BinObjFile);

      for (const object::SymbolRef &S : ElfObj->getDynamicSymbolIterators()) {
        uint32_t Flags = S.getFlags();
        // DO NOT insert to table if symbol was undefined
        if (Flags & llvm::object::SymbolRef::SF_Undefined)
          continue;

        // Note, we are at last resort and loading library based on a weak
        // symbol is allowed. Otherwise, the JIT will issue an unresolved
        // symbol error.
        //
        // There are other weak symbol kinds (marked as 'V') to denote
        // typeinfo and vtables. It is unclear whether we should load such
        // libraries or from which library we should resolve the symbol.
        // We seem to not have a way to differentiate it from the symbol API.

        llvm::Expected<StringRef> SymNameErr = S.getName();
        if (!SymNameErr) {
          cling::errs() << "Dyld::BuildBloomFilter: Failed to read symbol "
                        <<SymNameErr.get() << "\n";
          continue;
        }

        if (SymNameErr.get().empty())
          continue;

        ++SymbolsCount;
        symbols.push_back(SymNameErr.get());
      }
    }
    else if (BinObjFile->isCOFF()) { // On Windows, the symbols are present in COFF format.
      llvm::object::COFFObjectFile* CoffObj = cast<llvm::object::COFFObjectFile>(BinObjFile);

      // In COFF, the symbols are not present in the SymbolTable section
      // of the Object file. They are present in the ExportDirectory section.
      for (auto I=CoffObj->export_directory_begin(), 
                E=CoffObj->export_directory_end(); I != E; I = ++I) {
        // All the symbols are already flagged as exported. 
        // We cannot really ignore symbols based on flags as we do on unix.
        StringRef Name;
        if (I->getSymbolName(Name))
          continue;
        if (Name.empty())
          continue;

        ++SymbolsCount;
        symbols.push_back(Name);
      }
    }

    Lib->InitializeBloomFilter(SymbolsCount);

    if (!SymbolsCount) {
      if (DEBUG > 7)
        cling::errs() << "Dyld::BuildBloomFilter: No symbols!\n";
      return;
    }

    if (DEBUG > 7) {
      cling::errs() << "Dyld::BuildBloomFilter: Symbols:\n";
      for (auto it : symbols)
        cling::errs() << "Dyld::BuildBloomFilter" <<  "- " <<  it << "\n";
    }

    // Generate BloomFilter
    for (const auto &S : symbols) {
      if (m_UseHashTable)
        Lib->AddBloom(Lib->AddSymbol(S));
      else
        Lib->AddBloom(S);
    }
  }

  bool Dyld::ContainsSymbol(const LibraryPath* Lib,
                            StringRef mangledName,
                            unsigned IgnoreSymbolFlags /*= 0*/) const {
    const std::string library_filename = Lib->GetFullName();

    if (DEBUG > 7) {
      cling::errs() << "Dyld::ContainsSymbol: Find symbol: lib="
                    << library_filename << ", mangled="
                    << mangledName.str() << "\n";
    }

    auto ObjF = llvm::object::ObjectFile::createObjectFile(library_filename);
    if (llvm::Error Err = ObjF.takeError()) {
      if (DEBUG > 1) {
        std::string Message;
        handleAllErrors(std::move(Err), [&](llvm::ErrorInfoBase &EIB) {
            Message += EIB.message() + "; ";
          });
        cling::errs() << "Dyld::ContainsSymbol: Failed to read object file "
                      << library_filename << " Errors: " << Message << "\n";
      }
      return false;
    }

    llvm::object::ObjectFile *BinObjFile = ObjF.get().getBinary();

    uint32_t hashedMangle = GNUHash(mangledName);
    // Check for the gnu.hash section if ELF.
    // If the symbol doesn't exist, exit early.
    if (BinObjFile->isELF() &&
        !MayExistInElfObjectFile(BinObjFile, hashedMangle)) {
      if (DEBUG > 7)
        cling::errs() << "Dyld::ContainsSymbol: ELF BloomFilter: Skip symbol <" << mangledName.str() << ">.\n";
      return false;
    }

    if (m_UseBloomFilter) {
      // Use our bloom filters and create them if necessary.
      if (!Lib->hasBloomFilter())
        BuildBloomFilter(const_cast<LibraryPath*>(Lib), BinObjFile,
                         IgnoreSymbolFlags);

      // If the symbol does not exist, exit early. In case it may exist, iterate.
      if (!Lib->MayExistSymbol(hashedMangle)) {
        if (DEBUG > 7)
          cling::errs() << "Dyld::ContainsSymbol: BloomFilter: Skip symbol <" << mangledName.str() << ">.\n";
        return false;
      }
      if (DEBUG > 7)
        cling::errs() << "Dyld::ContainsSymbol: BloomFilter: Symbol <" << mangledName.str() << "> May exist."
                      << " Search for it. ";
    }

    if (m_UseHashTable) {
      bool result = Lib->ExistSymbol(mangledName);
      if (DEBUG > 7)
        cling::errs() << "Dyld::ContainsSymbol: HashTable: Symbol "
                      << (result ? "Exist" : "Not exist") << "\n";
      return result;
    }

    auto ForeachSymbol =
      [&library_filename](llvm::iterator_range<llvm::object::symbol_iterator> range,
         unsigned IgnoreSymbolFlags, llvm::StringRef mangledName) -> bool {
      for (const llvm::object::SymbolRef &S : range) {
        uint32_t Flags = S.getFlags();
        // Do not insert in the table symbols flagged to ignore.
        if (Flags & IgnoreSymbolFlags)
          continue;

        // Note, we are at last resort and loading library based on a weak
        // symbol is allowed. Otherwise, the JIT will issue an unresolved
        // symbol error.
        //
        // There are other weak symbol kinds (marked as 'V') to denote
        // typeinfo and vtables. It is unclear whether we should load such
        // libraries or from which library we should resolve the symbol.
        // We seem to not have a way to differentiate it from the symbol API.

        llvm::Expected<llvm::StringRef> SymNameErr = S.getName();
        if (!SymNameErr) {
          cling::errs() << "Dyld::ContainsSymbol: Failed to read symbol "
                        << mangledName.str() << "\n";
          continue;
        }

        if (SymNameErr.get().empty())
          continue;

        if (SymNameErr.get() == mangledName) {
          if (DEBUG > 1) {
            cling::errs() << "Dyld::ContainsSymbol: Symbol "
                          << mangledName.str() << " found in "
                          << library_filename << "\n";
            return true;
          }
        }
      }
      return false;
    };

    // If no hash symbol then iterate to detect symbol
    // We Iterate only if BloomFilter and/or SymbolHashTable are not supported.

    if (DEBUG > 7)
      cling::errs() << "Dyld::ContainsSymbol: Iterate all for <"
                    << mangledName.str() << ">";

    // Symbol may exist. Iterate.
    if (ForeachSymbol(BinObjFile->symbols(), IgnoreSymbolFlags, mangledName)) {
      if (DEBUG > 7)
        cling::errs() << " -> found.\n";
      return true;
    }


    if (!BinObjFile->isELF()) {
      if (DEBUG > 7)
        cling::errs() << " -> not found.\n";
      return false;
    }

    // ELF file format has .dynstr section for the dynamic symbol table.
    const auto *ElfObj =
      llvm::cast<llvm::object::ELFObjectFileBase>(BinObjFile);

    bool result = ForeachSymbol(ElfObj->getDynamicSymbolIterators(),
                           IgnoreSymbolFlags, mangledName);
    if (DEBUG > 7)
        cling::errs() << (result ? " -> found.\n" : " -> not found.\n");
    return result;
  }

  bool Dyld::ShouldPermanentlyIgnore(StringRef FileName) const {
    assert(!m_ExecutableFormat.empty() && "Failed to find the object format!");

    if (!cling::DynamicLibraryManager::isSharedLibrary(FileName))
      return true;

    // No need to check linked libraries, as this function is only invoked
    // for symbols that cannot be found (neither by dlsym nor in the JIT).
    if (m_DynamicLibraryManager.isLibraryLoaded(FileName))
      return true;


    auto ObjF = llvm::object::ObjectFile::createObjectFile(FileName);
    if (!ObjF) {
      if (DEBUG > 1)
        cling::errs() << "[DyLD] Failed to read object file "
                      << FileName << "\n";
      return true;
    }

    llvm::object::ObjectFile *file = ObjF.get().getBinary();

    if (DEBUG > 1)
      cling::errs() << "Current executable format: " << m_ExecutableFormat
                    << ". Executable format of " << FileName << " : "
                    << file->getFileFormatName() << "\n";

    // Ignore libraries with different format than the executing one.
    if (m_ExecutableFormat != file->getFileFormatName())
      return true;

    if (llvm::isa<llvm::object::ELFObjectFileBase>(*file)) {
      for (auto S : file->sections()) {
        llvm::StringRef name;
        S.getName(name);
        if (name == ".text") {
          // Check if the library has only debug symbols, usually when
          // stripped with objcopy --only-keep-debug. This check is done by
          // reading the manual of objcopy and inspection of stripped with
          // objcopy libraries.
          auto SecRef = static_cast<llvm::object::ELFSectionRef&>(S);
          if (SecRef.getType() == llvm::ELF::SHT_NOBITS)
            return true;

          return (SecRef.getFlags() & llvm::ELF::SHF_ALLOC) == 0;
        }
      }
      return true;
    }

    //FIXME: Handle osx using isStripped after upgrading to llvm9.

    return m_ShouldPermanentlyIgnoreCallback(FileName);
  }

  void Dyld::dumpDebugInfo() const {
    cling::errs() << "Dyld: m_BasePaths:\n";
    cling::errs() << "---\n";
    size_t x = 0;
    for (auto const &item : m_BasePaths.m_Paths) {
      cling::errs() << "Dyld: - m_BasePaths[" << x++ << "]:"
                << &item << ": " << item << "\n";
    }
    cling::errs() << "---\n";
    x = 0;
    for (auto const &item : m_Libraries.GetLibraries()) {
      cling::errs() << "Dyld: - m_Libraries[" << x++ << "]:"
                    << &item << ": " << item->m_Path << ", "
                    << item->m_LibName << "\n";
    }
    x = 0;
    for (auto const &item : m_SysLibraries.GetLibraries()) {
      cling::errs() << "Dyld: - m_SysLibraries[" << x++ << "]:"
                    << &item << ": " << item->m_Path << ", "
                    << item->m_LibName << "\n";
    }
  }

  std::string Dyld::searchLibrariesForSymbol(StringRef mangledName,
                                             bool searchSystem/* = true*/) {
    assert(!llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(mangledName) &&
           "Library already loaded, please use dlsym!");
    assert(!mangledName.empty());

    using namespace llvm::sys::path;
    using namespace llvm::sys::fs;

    if (m_FirstRun) {
      if (DEBUG > 7) {
        cling::errs() << "Dyld::searchLibrariesForSymbol:" << mangledName.str() <<
          ", searchSystem=" << (searchSystem ? "true" : "false") << ", FirstRun(user)... scanning\n";
      }

      if (DEBUG > 7) {
        cling::errs() << "Dyld::searchLibrariesForSymbol: Before first ScanForLibraries\n";
        dumpDebugInfo();
      }

      ScanForLibraries(/* SearchSystemLibraries= */ false);
      m_FirstRun = false;

      if (DEBUG > 7) {
        cling::errs() << "Dyld::searchLibrariesForSymbol: After first ScanForLibraries\n";
        dumpDebugInfo();
      }
    }

    if (m_QueriedLibraries.size() > 0) {
      // Last call we were asked if a library contains a symbol. Usually, the
      // caller wants to load this library. Check if was loaded and remove it
      // from our lists of not-yet-loaded libs.

      if (DEBUG > 7) {
        cling::errs() << "Dyld::ResolveSymbol: m_QueriedLibraries:\n";
        size_t x = 0;
        for (auto item : m_QueriedLibraries.GetLibraries()) {
          cling::errs() << "Dyld::ResolveSymbol - [" << x++ << "]:"
                        << &item << ": " << item->GetFullName() << "\n";
        }
      }

      for (const LibraryPath* P : m_QueriedLibraries.GetLibraries()) {
        const std::string LibName = P->GetFullName();
        if (!m_DynamicLibraryManager.isLibraryLoaded(LibName))
          continue;

        m_Libraries.UnregisterLib(*P);
        m_SysLibraries.UnregisterLib(*P);
      }
      // TODO:  m_QueriedLibraries.clear ?
    }

    // Iterate over files under this path. We want to get each ".so" files
    for (const LibraryPath* P : m_Libraries.GetLibraries()) {
      if (ContainsSymbol(P, mangledName, /*ignore*/
                         llvm::object::SymbolRef::SF_Undefined)) {
        if (!m_QueriedLibraries.HasRegisteredLib(*P))
          m_QueriedLibraries.RegisterLib(*P);

        if (DEBUG > 7)
          cling::errs() << "Dyld::ResolveSymbol: Search found match in [user lib]: "
                        << P->GetFullName() << "!\n";

        return P->GetFullName();
      }
    }

    if (!searchSystem)
      return "";

    if (DEBUG > 7)
      cling::errs() << "Dyld::searchLibrariesForSymbol: SearchSystem!!!\n";

    // Lookup in non-system libraries failed. Expand the search to the system.
    if (m_FirstRunSysLib) {
      if (DEBUG > 7) {
        cling::errs() << "Dyld::searchLibrariesForSymbol:" << mangledName.str() <<
          ", searchSystem=" << (searchSystem ? "true" : "false") << ", FirstRun(system)... scanning\n";
      }

      if (DEBUG > 7) {
        cling::errs() << "Dyld::searchLibrariesForSymbol: Before first system ScanForLibraries\n";
        dumpDebugInfo();
      }

      ScanForLibraries(/* SearchSystemLibraries= */ true);
      m_FirstRunSysLib = false;

      if (DEBUG > 7) {
        cling::errs() << "Dyld::searchLibrariesForSymbol: After first system ScanForLibraries\n";
        dumpDebugInfo();
      }
    }

    for (const LibraryPath* P : m_SysLibraries.GetLibraries()) {
      if (ContainsSymbol(P, mangledName, /*ignore*/
                         llvm::object::SymbolRef::SF_Undefined |
                         llvm::object::SymbolRef::SF_Weak)) {
        if (!m_QueriedLibraries.HasRegisteredLib(*P))
          m_QueriedLibraries.RegisterLib(*P);

        if (DEBUG > 7)
          cling::errs() << "Dyld::ResolveSymbol: Search found match in [system lib]: "
                        << P->GetFullName() << "!\n";

        return P->GetFullName();
      }
    }

    if (DEBUG > 7)
      cling::errs() << "Dyld::ResolveSymbol: Search found no match!\n";

    return ""; // Search found no match.
  }

  DynamicLibraryManager::~DynamicLibraryManager() {
    static_assert(sizeof(Dyld) > 0, "Incomplete type");
    delete m_Dyld;
  }

  void DynamicLibraryManager::initializeDyld(
                 std::function<bool(llvm::StringRef)> shouldPermanentlyIgnore) {
     //assert(!m_Dyld && "Already initialized!");
    if (m_Dyld)
      delete m_Dyld;

    std::string exeP = GetExecutablePath();
    auto ObjF =
      cantFail(llvm::object::ObjectFile::createObjectFile(exeP));

    m_Dyld = new Dyld(*this, shouldPermanentlyIgnore,
                      ObjF.getBinary()->getFileFormatName());
  }

  std::string
  DynamicLibraryManager::searchLibrariesForSymbol(StringRef mangledName,
                                           bool searchSystem/* = true*/) const {
    assert(m_Dyld && "Must call initialize dyld before!");
    return m_Dyld->searchLibrariesForSymbol(mangledName, searchSystem);
  }

  std::string DynamicLibraryManager::getSymbolLocation(void *func) {
#if defined(__CYGWIN__) && defined(__GNUC__)
    return {};
#elif defined(_WIN32)
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery (func, &mbi, sizeof (mbi)))
      return {};

    HMODULE hMod = (HMODULE) mbi.AllocationBase;
    char moduleName[MAX_PATH];

    if (!GetModuleFileNameA (hMod, moduleName, sizeof (moduleName)))
      return {};

    return cached_realpath(moduleName);

#else
    // assume we have  defined HAVE_DLFCN_H and HAVE_DLADDR
    Dl_info info;
    if (dladdr((void*)func, &info) == 0) {
      // Not in a known shared library, let's give up
      return {};
    } else {
      std::string result = cached_realpath(info.dli_fname);
      if (!result.empty())
        return result;

      // Else absolute path. For all we know that's a binary.
      // Some people have dictionaries in binaries, this is how we find their
      // path: (see also https://stackoverflow.com/a/1024937/6182509)
# if defined(__APPLE__)
      char buf[PATH_MAX] = { 0 };
      uint32_t bufsize = sizeof(buf);
      if (_NSGetExecutablePath(buf, &bufsize) >= 0)
        return cached_realpath(buf);
      return cached_realpath(info.dli_fname);
# elif defined(LLVM_ON_UNIX)
      char buf[PATH_MAX] = { 0 };
      // Cross our fingers that /proc/self/exe exists.
      if (readlink("/proc/self/exe", buf, sizeof(buf)) > 0)
        return cached_realpath(buf);
      std::string pipeCmd = std::string("which \"") + info.dli_fname + "\"";
      FILE* pipe = popen(pipeCmd.c_str(), "r");
      if (!pipe)
        return cached_realpath(info.dli_fname);
      while (fgets(buf, sizeof(buf), pipe))
         result += buf;

      pclose(pipe);
      return cached_realpath(result);
# else
#  error "Unsupported platform."
# endif
      return {};
   }
#endif
  }

} // namespace cling
