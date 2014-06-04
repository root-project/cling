
#include "FSUtils.h"

#include "readtags.h"

#include "cling/TagsExtension/CtagsWrapper.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace cling {
  struct TagFileInternals{
    tagFile* tf;
    tagFileInfo tfi;
  };

  CtagsFileWrapper::CtagsFileWrapper(std::string path, bool recurse, bool fileP)
    :TagFileWrapper(path), m_Tagfile(new TagFileInternals()) {
//      llvm::errs()<<path<<'\n';
//    m_Tagfile = new TagFileInternals();
    if (fileP) {
      generate(path);
      read();
      return;
    }
    if (recurse) {
      std::vector<std::string> list;
      llvm::error_code ec;
      llvm::sys::fs::recursive_directory_iterator rdit(path, ec);
      while (rdit != decltype(rdit)()) {
        auto entry = *rdit;

        if (llvm::sys::fs::is_regular_file (entry.path())
                && isHeaderFile(entry.path())) {
          //llvm::outs()<<entry.path()<<"\n";
          list.push_back(entry.path());
        }
        rdit.increment(ec);
      }
      generate(list,path);
      read();
    }
    else {
      llvm::error_code ec;
      llvm::sys::fs::directory_iterator dit(path, ec);
      std::vector<std::string> list;
      while (dit != decltype(dit)()){// !=end iterator
        auto entry = *dit;
        if (llvm::sys::fs::is_regular_file (entry.path()))
          //llvm::outs()<<entry.path()<<"\n";
          list.push_back(entry.path());
        dit.increment(ec);
      }
      ///auto pair=splitPath(path);
      ///TODO Preprocess the files in list and generate tags for them
    }
  }

  std::map<std::string, TagFileWrapper::LookupResult>
  CtagsFileWrapper::match(std::string name, bool partialMatch){
    std::map<std::string,LookupResult> map;
    tagEntry entry;
    int options = TAG_OBSERVECASE | (partialMatch?TAG_PARTIALMATCH:TAG_FULLMATCH);
    
    tagResult result = tagsFind(m_Tagfile->tf, &entry, name.c_str(), options);
    
    while (result==TagSuccess){
      LookupResult r;
      r.name = entry.name;
      r.kind = entry.kind;
      map[entry.file] = r;
      result=tagsFindNext(m_Tagfile->tf, &entry);
    }
    
    return map;
  }

  void CtagsFileWrapper::generate(std::string file) {
    m_Tagpath = generateTagPath();
    m_Tagfilename = pathToFileName(file);

    if (!needToGenerate(m_Tagpath,m_Tagfilename, file)){
      m_Generated=false;
      return;
    }
    std::string cmd = "ctags --language-force=c++ -f "
      + m_Tagpath + m_Tagfilename + " " + file;
//    llvm::errs()<<cmd<<"\n";
    system(cmd.c_str());
  }

  //no more than `arglimit` arguments in a single invocation
  void CtagsFileWrapper::generate(const std::vector<std::string>& paths,
                                   std::string dirpath) {
    std::string concat;
    m_Tagpath = generateTagPath();
    m_Tagfilename = pathToFileName(dirpath);

    if (!needToGenerate(m_Tagpath, m_Tagfilename, dirpath)){
      m_Generated = false;
      return;
    }
    auto it = paths.begin(), end = paths.end();
    while (it != end) {
      concat += (*it + " ");
      it++;
    }

    //TODO: Convert these to twine
    std::string filename = " -f " + m_Tagpath+m_Tagfilename + " ";
    std::string lang = " --language-force=c++ ";
    std::string sorted = " --sort=yes ";
    std::string append = " -a ";
    std::string cmd = "ctags "+ append + lang + filename + sorted + concat;

//        llvm::errs()<<cmd<<"\n";

    std::system(cmd.c_str());
    m_Generated = true;
  }
  
  void CtagsFileWrapper::read() {
    m_Tagfile->tf
      = tagsOpen((m_Tagpath + m_Tagfilename).c_str(), &(m_Tagfile->tfi));

    //std::cout<<"File "<<tagpath+tagfilename<<" read.\n";
    if (m_Tagfile->tfi.status.opened == false)
        m_Validfile = false;
    else
        m_Validfile = true;
  }
}
