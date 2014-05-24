#ifndef CLING_FS_UTILS_H
#define CLING_FS_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace cling {
  std::string pathToFileName(std::string path);

  bool fileIsNewer(std::string path, std::string dir);

  bool needToGenerate(std::string tagpath, std::string filename,
                      std::string dirpath);

  std::string generateTagPath();

  bool isHeaderFile(llvm::StringRef str);

  std::pair<std::string,std::string> splitPath(std::string path);

} // end namespace cling
#endif // CLING_FS_UTILS_H
