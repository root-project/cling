#include "Wrapper.h"
namespace cling {
  struct TagFileInternals;
  ///\brief Implements tag operations for Ctags
  class CtagsFileWrapper : public TagFileWrapper {
  public:
    CtagsFileWrapper(std::string path, bool recurse = true, bool fileP = false);

    ~CtagsFileWrapper(){}

    std::map<std::string,LookupResult>
    match(std::string name, bool partialMatch = false);

    bool newFile() const { return m_Generated; }

    bool validFile() const { return m_Validfile; }

  private:
    void generate(const std::vector<std::string>& cmd,
                  std::string tagfile = "adhoc");
    void generate(std::string file);

    void read();

    TagFileInternals* m_Tagfile;
    std::string m_Tagfilename;
    std::string m_Tagpath;
    bool m_Generated;
    bool m_Validfile;
  };
}//end namespace cling
