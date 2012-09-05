OPTION("<input>", INPUT, Input, INVALID, INVALID, 0, 0, 0, 0)
OPTION("<unknown>", UNKNOWN, Unknown, INVALID, INVALID, 0, 0, 0, 0)
// Re-implement to forward to our help
OPTION("--help", _help, Flag, INVALID, help, 0, 0, 0, 0)
OPTION("--metastr=", _metastr_EQ, Joined, INVALID, INVALID, 0, 0,
       "Set the meta command tag, default '.'", 0)
OPTION("--metastr", _metastr, Separate, INVALID, INVALID, 0, 0,
       "Set the meta command tag, default '.'", 0)
OPTION("--nologo", _nologo, Flag, INVALID, INVALID, 0, 0,
       "Do not show startup-banner", 0)
OPTION("--version", _version, Flag, INVALID, version, 0, 0, 0, 0)
OPTION("-L", L, JoinedOrSeparate, INVALID, INVALID, 0, 0,
       "Add directory to library search path", "<directory>")
OPTION("-help", help, Flag, INVALID, INVALID, 0, 0,
       "Print this help text", 0)
OPTION("-l", l, JoinedOrSeparate, INVALID, INVALID, 0, 0,
       "Load a library before prompt", "<library>")
OPTION("-version", version, Flag, INVALID, INVALID, 0, 0,
       "Print the compiler version", 0)
OPTION("-v", v, Flag, INVALID, INVALID, 0, 0,
       "Enable verbose output", 0)
