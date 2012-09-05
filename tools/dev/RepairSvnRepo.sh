#!/bin/bash

################################################################################
# Simple script that helps to heal the svn update command, which overwrites    #
# the current local rev number of cling's repo when inline in llvm source tree.#
# It takes the revision that it thinks is the current, the correct revision    #
# and optional path. If no path provided it uses the current dir.              #
################################################################################
#                                                                              #
#                Author: Vassil Vassilev (vvasilev@cern.ch)                    #
#                                                                              #
################################################################################



if [ "$1" = "--help" -o -z "$1" ]; then
    echo "Usage sudo $0 BrokenRev CorrectRev [Path]"
    exit
fi

if [ -z $3 ]; then
    Folder="."
else
    Folder=$3
fi

if [ -z "$1" -o -z "$2" ]; then
    echo "Missing arguments. Try --help"
    exit
fi

FilesToFix=$(find $Folder -iname entries | grep .svn)

for file in $FilesToFix; 
do 
    echo "Changing go+w: $file";
    chmod go+w "$file"
    echo "Replacing $1 with $2: $file";
    sed -i.bak -e "s/$1/$2/" "$file"
    echo "Changing go-w: $file";
    chmod go-w "$file"
done

