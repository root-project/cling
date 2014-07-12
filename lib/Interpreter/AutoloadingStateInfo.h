#ifndef CLING_AUTOLOADING_STATEINFO
#define CLING_AUTOLOADING_STATEINFO
#include <vector>
#include <map>

namespace clang {
  class Decl;
}

namespace cling {
  class AutoloadingStateInfo {
  public:
    struct FileInfo {
      FileInfo():Included(false){}
      bool Included;
      std::vector<clang::Decl*> Decls;
    };

    // The key is the Unique File ID obtained from the source manager.
    std::map<unsigned,FileInfo> m_Map;
  };
} // end namespace cling
#endif
