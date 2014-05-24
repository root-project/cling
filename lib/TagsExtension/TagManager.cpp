#include "cling/TagsExtension/TagManager.h"

#include "cling/TagsExtension/CtagsWrapper.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace cling {

  TagManager::TagManager() {}
  void TagManager::AddTagFile(std::string path, bool recurse) {
    bool fileP = false;
    if (llvm::sys::fs::is_regular_file(path)) {
      fileP = true;
    }
    if (llvm::sys::path::is_relative(path)) {
      llvm::SmallString<100> str(path.data());
      llvm::error_code ec = llvm::sys::fs::make_absolute(str);
      if (ec != llvm::errc::success)
        llvm::errs()<<"Can't deduce absolute path.\n";
      else
        path = str.c_str();
    }

    TagFileWrapper* tf = new CtagsFileWrapper(path, recurse, fileP);
    if (!tf->validFile()) {
      llvm::errs() << "Reading Tag File: " << path << " failed.\n";
      return;
    }
    bool eq = false;
    for (auto& t : m_Tags) {
      if (*t == *tf ) {
        eq = true;
        break;
      }
    }
    if (!eq) {
      m_Tags.push_back(tf);
    }
  }

  TagManager::TableType::iterator TagManager::begin(std::string name) {
    m_Table.erase(name);
    for (auto& t : m_Tags){
      for (auto match : t->match(name, true)){
        LookupInfo l(match.first, match.second.name, match.second.kind);
        m_Table.insert({name, l});
      }
    }
    auto r = m_Table.equal_range(name);
    return r.first;
  }

  TagManager::TableType::iterator TagManager::end(std::string name) {
    auto r = m_Table.equal_range(name);
    return r.second;
  }

  TagManager::LookupInfo::LookupInfo (std::string h, std::string n,
                                      std::string t)
    : header(h), name(n), type(t){}

  TagManager::~TagManager() {
    for (auto& tag : m_Tags )
      delete tag;
  }
} //end namespace cling
