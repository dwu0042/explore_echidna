#!/bin/bash

# fragile script for converting the flow-based "temponet" networks from VALT to equivalent reciprocal weights

cat $1 | awk '{ print $1, $2, 1/$3 }' > ${1/temponet/flownet}