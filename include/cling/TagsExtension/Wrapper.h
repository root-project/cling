#ifndef CLING_CTAGS_WRAPPER_H
#define CLING_CTAGS_WRAPPER_H
// #include "readtags.h"
#include <vector>
#include <cstdlib>
#include <map>
#include <string>

namespace cling {
  ///\brief Different tag generating systems must inherit from TagFileWrapper
  /// An object of a derived class represents a single tag file,
  /// which may be generated from multiple header/source files
  class TagFileWrapper {
  public:
    TagFileWrapper(std::string Path):m_Path(Path){}

    struct LookupResult{
      std::string name;
      std::string kind;
    };
    ///\brief Returns a map which associates a name with header files
    /// and type of the name
    /// When partialMatch is true, name can be a prefix of the values returned
    virtual std::map<std::string,LookupResult>
            match(std::string name, bool partialMatch=false)=0;

    ///\brief True if the file was generated and not already present
    virtual bool newFile()=0;

    ///\brief True if the file is in a valid state
    virtual bool validFile()=0;

    virtual ~TagFileWrapper(){}

    bool operator==(const TagFileWrapper& t) {return m_Path==t.m_Path;}
  private:
    std::string m_Path;
  };
} //end namespace cling


#endif
