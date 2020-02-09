#!/bin/bash
workdir="`dirname \"$0\"`"
workdir_abs="`realpath \"$workdir\"`"
./grasp_provider.py 2>/dev/null

