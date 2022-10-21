#!/bin/sh
# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

#SHA1 of the first commit compatible with the current model
commit=b7d25ac
./download_model.sh $commit

echo "Updating build configuration files for lpcnet, please wait...."

autoreconf -isf
