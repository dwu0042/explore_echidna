#!/bin/bash
sed "s/,\s*/,/g; s/;\s*/ /g" < $1 > "$1.lgl"