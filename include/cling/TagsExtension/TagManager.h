#ifndef CLING_TAG_MANAGER_H
#define CLING_TAG_MANAGER_H
#include "cling/Interpreter/Interpreter.h"
#include "cling/TagsExtension/Wrapper.h"

#include <map>
#include <vector>

namespace cling {
  ///\brief Class for managing all the available tag files
  class TagManager {
  public:
    TagManager();
    
    ///\brief Information from a tag file lookup
    class LookupInfo{
    public:
      LookupInfo(std::string h, std::string n, std::string t);
      std::string header;
      std::string name;
      std::string type;
    };

    ///\brief Add a new path from which a single tag file is generated
    /// If recurse is true, path is recursively scanned but not preprocessed
    /// else only the files directly in path are preprocessed and tagged
    ///
    void AddTagFile(std::string path, bool recurse=true);

    std::size_t size() { return m_Tags.size(); }

    typedef std::multimap<std::string,LookupInfo> TableType;
    TableType::iterator begin(std::string name);
    TableType::iterator end(std::string name);

    ~TagManager();
  private:
    std::vector<TagFileWrapper*> m_Tags;
    TableType m_Table;
  };
}// end namespace cling
#endif // CLING_TAG_MANAGER_H
