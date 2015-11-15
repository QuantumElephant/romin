#!/bin/bash
echo "Cleaning code in ${PWD} and subdirectories."
# split output of find at newlines.
IFS=$'\n'
# send all relevant files to the code cleaner
find romin *.* scripts tools | \
    egrep "(^scripts/)|(\.cpp$)|(\.h$)|(\.in$)|(\.sh$)|(\.py$)|(\.pyx$)|(\.pxd$)|(\.txt$)|(\.conf$)|(.gitignore$)" | \
    xargs ./tools/codecleaner.py
# Remove files that "contaminate" the source tree. The cmake build dir should be included
# here.
rm -vfr build
rm -vf denspart/cppext.so
rm -vf denspart/cppext.cpp
