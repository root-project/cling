The patches are now tracked in the git repositories

  http://root.cern.ch/git/llvm.git (mirror of llvm's git)
  http://root.cern.ch/git/clang.git (mirror of clang's git)

To build cling check out the tag called "cling-patches-rREV" from both repos,
where REV comes from LastKnownGoodLLVMSVNRevision.txt, for instance:

  $ cd src; git checkout cling-patches-r191429
  $ cd tools/clang; git checkout cling-patches-r191429

See http://cern.ch/cling for build instructions.
