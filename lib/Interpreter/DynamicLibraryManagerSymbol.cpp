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
#include "cling/Utils/Output.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <list>
#include <string>
#include <unordered_set>
#include <vector>

#include "llvm/Config/config.h" // Get configuration settings

#if defined(HAVE_DLFCN_H) && defined(HAVE_DLOPEN)
#include <dlfcn.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif // HAVE_UNISTD_H

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif // __APPLE__

#ifdef LLVM_ON_WIN32
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

  LibraryPath(const BasePath& Path, const std::string& LibName)
    : m_Path(Path), m_LibName(LibName) { }

  bool operator==(const LibraryPath &other) const {
    return (&m_Path == &other.m_Path || m_Path == other.m_Path) &&
      m_LibName == other.m_LibName;
  }

  std::string GetFullName() const {
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
    assert(isBloomFilterEmpty() && "Cannot re-initialize non-empty filter!");
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

  void RegisterLib(const LibraryPath& Lib) {
    auto it = m_LibsH.insert(Lib);
    assert(it.second && "Already registered!");
    m_Libs.push_back(&*it.first);
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

    bool Contains (const std::string& Path) {
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
  std::vector<LibraryPath> m_QueriedLibraries;

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
  bool ContainsSymbol(const LibraryPath* Lib, const std::string &mangledName,
                      unsigned IgnoreSymbolFlags = 0) const;

protected:
  Dyld(const cling::DynamicLibraryManager &DLM)
    : m_DynamicLibraryManager(DLM) { }

  ~Dyld() = default;

public:
  static Dyld& getInstance(const cling::DynamicLibraryManager &DLM) {
    static Dyld instance(DLM);

#ifndef NDEBUG
    auto &NewSearchPaths = DLM.getSearchPaths();
    auto &OldSearchPaths = instance.m_DynamicLibraryManager.getSearchPaths();
    // FIXME: Move the Dyld logic to the cling::DynamicLibraryManager itself!
    assert(std::equal(OldSearchPaths.begin(), OldSearchPaths.end(),
                      NewSearchPaths.begin()) && "Path was added/removed!");
#endif

    return instance;
  }

  // delete copy and move constructors and assign operators
  Dyld(Dyld const&) = delete;
  Dyld(Dyld&&) = delete;
  Dyld& operator=(Dyld const&) = delete;
  Dyld& operator=(Dyld &&) = delete;

  std::string searchLibrariesForSymbol(const std::string& mangledName,
                                       bool searchSystem);
};


static bool s_IsDyldInitialized = false;
static std::function<bool(llvm::StringRef)> s_ShouldPermanentlyIgnoreCallback;


static std::string getRealPath(llvm::StringRef path) {
  llvm::SmallString<512> realPath;
  llvm::sys::fs::real_path(path, realPath, /*expandTilde*/true);
  return realPath.str().str();
}

static llvm::StringRef s_ExecutableFormat;

static bool shouldPermanentlyIgnore(const std::string& FileName,
                            const cling::DynamicLibraryManager& dyLibManager) {
  assert(FileName == getRealPath(FileName));
  assert(!s_ExecutableFormat.empty() && "Failed to find the object format!");

  if (llvm::sys::fs::is_directory(FileName))
    return true;

  if (!cling::DynamicLibraryManager::isSharedLibrary(FileName))
    return true;

  // No need to check linked libraries, as this function is only invoked
  // for symbols that cannot be found (neither by dlsym nor in the JIT).
  if (dyLibManager.isLibraryLoaded(FileName.c_str()))
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
     cling::errs() << "Current executable format: " << s_ExecutableFormat
                   << ". Executable format of " << FileName << " : "
                   << file->getFileFormatName() << "\n";

  // Ignore libraries with different format than the executing one.
  if (s_ExecutableFormat != file->getFileFormatName())
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

  return s_ShouldPermanentlyIgnoreCallback(FileName);
}

void Dyld::ScanForLibraries(bool searchSystemLibraries/* = false*/) {

  // #ifndef NDEBUG
  //   if (!m_FirstRun && !m_FirstRunSysLib)
  //     assert(0 && "Already initialized");
  //   if (m_FirstRun && !m_Libraries->size())
  //     assert(0 && "Not initialized but m_Libraries is non-empty!");
  //   // assert((m_FirstRun || m_FirstRunSysLib) && (m_Libraries->size() ||
  //             m_SysLibraries.size())
  //   //        && "Already scanned and initialized!");
  // #endif

  const auto &searchPaths = m_DynamicLibraryManager.getSearchPaths();
  for (const cling::DynamicLibraryManager::SearchPathInfo &Info : searchPaths) {
    if (Info.IsUser || searchSystemLibraries) {
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
      // Paths = /lib/1 : /lib/2 : /lib/3

      // m_BasePaths = ["/lib/1", "/lib/3", "/system/lib"]
      // m_*Libraries  = [<0,"1.so">, <1,"1.so">, <2,"s.so">, <1,"3.so">]
      std::string RealPath = getRealPath(Info.Path);
      llvm::StringRef DirPath(RealPath);

      if (!llvm::sys::fs::is_directory(DirPath) || DirPath.empty())
        continue;

      // Already searched?
      bool WasInserted;
      m_BasePaths.RegisterBasePath(RealPath, &WasInserted);

      if (!WasInserted)
        continue;

      std::error_code EC;
      for (llvm::sys::fs::directory_iterator DirIt(DirPath, EC), DirEnd;
           DirIt != DirEnd && !EC; DirIt.increment(EC)) {

        // FIXME: Use a StringRef here!
        std::string FileName = getRealPath(DirIt->path());
        assert(!llvm::sys::fs::is_symlink_file(FileName));

        if (shouldPermanentlyIgnore(FileName, m_DynamicLibraryManager))
          continue;

        std::string FileRealPath = llvm::sys::path::parent_path(FileName);
        FileName = llvm::sys::path::filename(FileName);
        const BasePath& BaseP = m_BasePaths.RegisterBasePath(FileRealPath);
        LibraryPath LibPath(BaseP, FileName);
        if (m_SysLibraries.HasRegisteredLib(LibPath) ||
            m_Libraries.HasRegisteredLib(LibPath))
          continue;

        if (searchSystemLibraries)
          m_SysLibraries.RegisterLib(LibPath);
        else
          m_Libraries.RegisterLib(LibPath);
      }
    }
  }
}

void Dyld::BuildBloomFilter(LibraryPath* Lib,
                            llvm::object::ObjectFile *BinObjFile,
                            unsigned IgnoreSymbolFlags /*= 0*/) const {
  assert(m_UseBloomFilter && "Bloom filter is disabled");
  assert(Lib->hasBloomFilter() && "Already built!");

  using namespace llvm;
  using namespace llvm::object;

  // If BloomFilter is empty then build it.
  // Count Symbols and generate BloomFilter
  uint32_t SymbolsCount = 0;
  std::list<std::string> symbols;
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

  Lib->InitializeBloomFilter(SymbolsCount);
  // Generate BloomFilter
  for (const auto &S : symbols) {
    if (m_UseHashTable)
      Lib->AddBloom(Lib->AddSymbol(S));
    else
      Lib->AddBloom(S);
  }
}


static llvm::StringRef GetGnuHashSection(llvm::object::ObjectFile *file) {
  for (auto S : file->sections()) {
    llvm::StringRef name;
    S.getName(name);
    if (name == ".gnu.hash") {
      llvm::StringRef content;
      S.getContents(content);
      return content;
    }
  }
  return "";
}

/// Bloom filter in a stohastic data structure which can tell us if a symbol
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

  // LLVM9: soFile->makeTriple().is64Bit()
  const int bits = 8 * soFile->getBytesInAddress();

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

bool Dyld::ContainsSymbol(const LibraryPath* Lib,
                          const std::string &mangledName,
                          unsigned IgnoreSymbolFlags /*= 0*/) const {
  const std::string library_filename = Lib->GetFullName();

  if (DEBUG > 7) {
    cling::errs() << "Dyld::ContainsSymbol: Find symbol: lib="
                  << library_filename << ", mangled="
                  << mangledName << "\n";
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
  if (BinObjFile->isELF() && !MayExistInElfObjectFile(BinObjFile, hashedMangle))
    return false;

  if (m_UseBloomFilter) {
    // Use our bloom filters and create them if necessary.
    if (!Lib->hasBloomFilter())
      BuildBloomFilter(const_cast<LibraryPath*>(Lib), BinObjFile,
                       IgnoreSymbolFlags);

    // If the symbol does not exist, exit early. In case it may exist, iterate.
    if (!Lib->MayExistSymbol(hashedMangle)) {
      if (DEBUG > 7)
        cling::errs() << "Dyld::ContainsSymbol: BloomFilter: Skip symbol.\n";
      return false;
    }
    if (DEBUG > 7)
      cling::errs() << "Dyld::ContainsSymbol: BloomFilter: Symbol May exist."
                    << " Search for it.";
  }

  if (m_UseHashTable) {
    bool result = Lib->ExistSymbol(mangledName);
    if (DEBUG > 7)
      cling::errs() << "Dyld::ContainsSymbol: HashTable: Symbol "
                    << (result ? "Exist" : "Not exist") << "\n";
    return result;
  }

  // Symbol may exist. Iterate.

  // If no hash symbol then iterate to detect symbol
  // We Iterate only if BloomFilter and/or SymbolHashTable are not supported.
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
      cling::errs() << "Dyld::ContainsSymbol: Failed to read symbol "
                    << mangledName << "\n";
      continue;
    }

    if (SymNameErr.get().empty())
      continue;

    if (SymNameErr.get() == mangledName) {
      if (DEBUG > 1) {
        cling::errs() << "Dyld::ContainsSymbol: Symbol "
                      << mangledName << " found in "
                      << library_filename << "\n";
        return true;
      }
    }
  }

  if (!BinObjFile->isELF())
    return false;

  // ELF file format has .dynstr section for the dynamic symbol table.
  const auto *ElfObj = llvm::cast<llvm::object::ELFObjectFileBase>(BinObjFile);

  for (const llvm::object::SymbolRef &S : ElfObj->getDynamicSymbolIterators()) {
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

    llvm::Expected<llvm::StringRef> SymNameErr = S.getName();
    if (!SymNameErr) {
      cling::errs() << "Dyld::ContainsSymbol: Failed to read symbol "
                    << mangledName << "\n";
      continue;
    }

    if (SymNameErr.get().empty())
      continue;

    if (SymNameErr.get() == mangledName)
      return true;
  }
  return false;
}

std::string Dyld::searchLibrariesForSymbol(const std::string& mangledName,
                                           bool searchSystem/* = true*/) {
  assert(!llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(mangledName) &&
         "Library already loaded, please use dlsym!");
  assert(!mangledName.empty());
  using namespace llvm::sys::path;
  using namespace llvm::sys::fs;

  if (m_FirstRun) {
    ScanForLibraries(/* SearchSystemLibraries= */ false);
    m_FirstRun = false;
  }

  if (!m_QueriedLibraries.empty()) {
    // Last call we were asked if a library contains a symbol. Usually, the
    // caller wants to load this library. Check if was loaded and remove it
    // from our lists of not-yet-loaded libs.

    if (DEBUG > 7) {
      cling::errs() << "Dyld::ResolveSymbol: m_QueriedLibraries:\n";
      size_t x = 0;
      for (auto item : m_QueriedLibraries) {
        cling::errs() << "Dyld::ResolveSymbol - [" << x++ << "]:"
                      << &item << ": " << item.m_Path << ", "
                      << item.m_LibName << "\n";
      }
    }

    for (const LibraryPath& P : m_QueriedLibraries) {
      const std::string LibName = P.GetFullName();
      if (!m_DynamicLibraryManager.isLibraryLoaded(LibName))
        continue;

      m_Libraries.UnregisterLib(P);
      m_SysLibraries.UnregisterLib(P);
    }
  }

  // Iterate over files under this path. We want to get each ".so" files
  for (const LibraryPath* P : m_Libraries.GetLibraries()) {
    const std::string LibName = P->GetFullName();

    if (ContainsSymbol(P, mangledName, /*ignore*/
                       llvm::object::SymbolRef::SF_Undefined)) {
      m_QueriedLibraries.push_back(*P);
      return LibName;
    }
  }

  if (!searchSystem)
    return "";

  if (DEBUG > 7)
    cling::errs() << "Dyld::ResolveSymbol: SearchSystem!\n";

  // Lookup in non-system libraries failed. Expand the search to the system.
  if (m_FirstRunSysLib) {
    ScanForLibraries(/* SearchSystemLibraries= */ true);
    m_FirstRunSysLib = false;
  }

  for (const LibraryPath* P : m_SysLibraries.GetLibraries()) {
    const std::string LibName = P->GetFullName();
    if (ContainsSymbol(P, mangledName, /*ignore*/
                       llvm::object::SymbolRef::SF_Undefined |
                       llvm::object::SymbolRef::SF_Weak)) {
      m_QueriedLibraries.push_back(*P);
      return LibName;
    }
  }

  if (DEBUG > 7)
    cling::errs() << "Dyld::ResolveSymbol: Search found no match!\n";

  /*
    if (DEBUG > 7) {
    cling::errs() << "Dyld::ResolveSymbol: Structs after ResolveSymbol:\n");

    cling::errs() << "Dyld::ResolveSymbol - sPaths:\n");
    size_t x = 0;
    for (const auto &item : sPaths.GetPaths())
    cling::errs() << "Dyld::ResolveSymbol << [" x++ << "]: " << item << "\n";

    cling::errs() << "Dyld::ResolveSymbol - sLibs:\n");
    x = 0;
    for (const auto &item : sLibraries.GetLibraries())
    cling::errs() << "Dyld::ResolveSymbol ["
    << x++ << "]: " << item->Path << ", "
    << item->LibName << "\n";

    cling::errs() << "Dyld::ResolveSymbol - sSysLibs:");
    x = 0;
    for (const auto &item : sSysLibraries.GetLibraries())
    cling::errs() << "Dyld::ResolveSymbol ["
    << x++ << "]: " << item->Path << ", "
    << item->LibName << "\n";

    Info("Dyld::ResolveSymbol", "- sQueriedLibs:");
    x = 0;
    for (const auto &item : sQueriedLibraries)
    cling::errs() << "Dyld::ResolveSymbol ["
    << x++ << "]: " << item->Path << ", "
    << item->LibName << "\n";
    }
  */

  return ""; // Search found no match.
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
  void DynamicLibraryManager::initializeDyld(
           std::function<bool(llvm::StringRef)> shouldPermanentlyIgnore) const {
    assert(!s_IsDyldInitialized);
    s_ShouldPermanentlyIgnoreCallback = shouldPermanentlyIgnore;

    std::string exeP = GetExecutablePath();
    auto ObjF =
      cantFail(llvm::object::ObjectFile::createObjectFile(exeP));
       s_ExecutableFormat = ObjF.getBinary()->getFileFormatName();

    s_IsDyldInitialized = true;
  }

  std::string
  DynamicLibraryManager::searchLibrariesForSymbol(const std::string& mangledName,
                                           bool searchSystem/* = true*/) const {
    assert(s_IsDyldInitialized && "Must call initialize dyld before!");
    static Dyld& dyld = Dyld::getInstance(*this);
    return dyld.searchLibrariesForSymbol(mangledName, searchSystem);
  }

  std::string DynamicLibraryManager::getSymbolLocation(void *func) {
#if defined(__CYGWIN__) && defined(__GNUC__)
    return {};
#elif defined(LLVM_ON_WIN32)
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery (func, &mbi, sizeof (mbi)))
      return {};

    HMODULE hMod = (HMODULE) mbi.AllocationBase;
    char moduleName[MAX_PATH];

    if (!GetModuleFileNameA (hMod, moduleName, sizeof (moduleName)))
      return {};

    return getRealPath(moduleName);
#else
    // assume we have  defined HAVE_DLFCN_H and HAVE_DLADDR
    Dl_info info;
    if (dladdr((void*)func, &info) == 0) {
      // Not in a known shared library, let's give up
      return {};
    } else {
      if (strchr(info.dli_fname, '/'))
        return getRealPath(info.dli_fname);
      // Else absolute path. For all we know that's a binary.
      // Some people have dictionaries in binaries, this is how we find their
      // path: (see also https://stackoverflow.com/a/1024937/6182509)
# if defined(__APPLE__)
      char buf[PATH_MAX] = { 0 };
      uint32_t bufsize = sizeof(buf);
      if (_NSGetExecutablePath(buf, &bufsize) >= 0)
        return getRealPath(buf);
      return getRealPath(info.dli_fname);
# elif defined(LLVM_ON_UNIX)
      char buf[PATH_MAX] = { 0 };
      // Cross our fingers that /proc/self/exe exists.
      if (readlink("/proc/self/exe", buf, sizeof(buf)) > 0)
        return getRealPath(buf);
      std::string pipeCmd = std::string("which \"") + info.dli_fname + "\"";
      FILE* pipe = popen(pipeCmd.c_str(), "r");
      if (!pipe)
        return getRealPath(info.dli_fname);
      std::string result;
      while (fgets(buf, sizeof(buf), pipe))
         result += buf;

      pclose(pipe);
      return getRealPath(result);
# else
#  error "Unsupported platform."
# endif
      return {};
   }
#endif
  }

} // namespace cling
