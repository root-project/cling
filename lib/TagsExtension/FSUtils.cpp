#include "FSUtils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace cling {

  std::pair<std::string,std::string> splitPath(std::string path) {
    auto filename = llvm::sys::path::filename(path);
    llvm::SmallString<128> p(path.begin(),path.end());
    llvm::sys::path::remove_filename(p);
    return { p.c_str(),filename };
  }

  std::string pathToFileName(std::string path) {
    for(auto& c : path)
      if(c == '/')
        c = '_';
    return path;
  }

  bool fileIsNewer(std::string path, std::string dir){
    return true;//TODO Timestamp checks go here
  }

  bool needToGenerate(std::string tagpath, std::string filename,
                      std::string dirpath){
    if( llvm::sys::fs::exists(tagpath+filename)) {
      return false;
    }
    else if (!fileIsNewer(tagpath+filename,dirpath)){
      return false;
    }
    else {
      //std::cout<<"File doesn't exist";
      return true;
    }
  }

  //FIXME: Replace when this is available in llvm::sys::path
  std::string get_separator() {
  #ifdef LLVM_ON_WIN32
   const char preferred_separator = '\\';
  #else
   const char preferred_separator = '/';
  #endif
   return { preferred_separator };
  }

  std::string generateTagPath() {
    llvm::SmallString<30> home_ss;
    llvm::sys::path::home_directory(home_ss);
    std::string homedir = home_ss.c_str();
    if (homedir == "")
        homedir = ".";

    std::string tagdir = get_separator() + ".cling/";
    std::string result = homedir + tagdir;
    llvm::sys::fs::create_directory(result);
    return result;
  }

  bool isHeaderFile(llvm::StringRef str){
    return str.endswith(".h")
            || str.endswith(".hpp")
            || str.find("include") != llvm::StringRef::npos;
  }
}//end namespace cling
