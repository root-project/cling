#!/bin/bash

# Find cling and clang.

TESTDIR=`pwd | sed 's,/tools/clang/test.*$,,'`
cling_binary=$TESTDIR/bin/cling
if ! [ -x $cling_binary ]; then
    echo 'Cannot find cling binary!' >& 2
    exit 1
fi

invocation="$@"
file=""
cling_args="--nologo"
while ! [ "$1" = "" ] ; do
    case $1 in
        -print-file-name=include) `dirname $cling_binary`/clang $1; exit $?; ;;
        -cc1) ;;
        -triple) echo -e "Ignoring (probably invalid target): $invocation\n" >&2; exit 0; shift;;
        -internal-isystem) cling_args="$cling_args -Xclang -internal-isystem -Xclang $2"; shift;;
        -triple=*) echo -e "Ignoring (probably invalid target): $invocation\n" >&2; exit 0;;
        -target-abi) shift;;
        -I) cling_args="$cling_args -I $2"; shift ;;
        -ast-dump-filter) cling_args="$cling_args -Xclang -ast-dump-filter -Xclang $2"; shift ;;
        -o) if [ "$2" == "-" ] ; then shift; shift; fi;; # ignore cling does it by default
        -I*) cling_args="$cling_args $1" ;;
        -x) cling_args="$cling_args -Xclang $1 -Xclang $2"; shift ;;
        -*) cling_args="$cling_args -Xclang $1" ;;
        *) file=$1
    esac
    shift
done
if [ "$file" = "" ]; then
    echo "Cannot find file in $invocation" >&2
    exit 1
fi
langopt="-x c"
case ${file##*.} in
    c) langopt="-x c";;
    cxx | cpp) langopt="-x c++";;
    m) langopt="-x objective-c";;
    mm) langopt="-x objective-c++";;
esac
cling_args="$cling_args $langopt"

#sed 's,^\(// *RUN:.*\)| *FileCheck\b.*$,\1,' $file > ${file}_repl

testcase=".rawInput\n.storeState \"a\"\n";
testcase+=".L $file\n"
testcase+=".L $file\n"
clang_preprocessed=$(`dirname $cling_binary`/clang `echo "$invocation -E -CC" | sed 's,\-verify, ,g'`)
if ( echo $cling_args | grep '[-]verify' > /dev/null ) && ! ( echo "$clang_preprocessed" | grep -q 'expected-error' > /dev/null ); then
    testcase+=".U $file\n"
fi
testcase+=".compareState \"a\"\n"
testcase+=".q"

echo -e "\n\e[32mREAL INVOCATION:\e[0m" >&2
echo "$invocation" >&2
echo -e "\n" >&2

echo -e "\n\e[32mRUNNING:\e[0m" >&2
echo "echo -e '$testcase' | $cling_binary $cling_args" >&2
echo -e "\n" >&2

echo -e "\n\e[32mTODEBUG:\e[0m" >&2
echo "echo -e '$testcase' > '$TMP/testcase' && CLING_NOHISTORY=1 gdb $cling_binary || rm '$TMP/testcase'" >&2

echo -e "\n\e[32mGDB ARGS:\e[0m" >&2
echo "run $cling_args < '$TMP/testcase'" >&2
echo -e "\n" >&2

#Known failures: Sema/warn-unused-function.c < We cannot know whether we need a function or not
#
! ( echo -e $testcase | $cling_binary $cling_args  2>& 1 |  tee /dev/stderr  | grep '^Differences in the ' )
