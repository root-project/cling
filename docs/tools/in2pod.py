#!/usr/bin/env python

# A tool to parse cling.pod.in and generate cling.pod dynamically

from __future__ import print_function
import subprocess
import sys
import os
import inspect

SCRIPT_DIR=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
cling_binary=sys.argv[1]

cmd=subprocess.Popen(["echo .help | %s --nologo" %(cling_binary)], stdout=subprocess.PIPE, shell=True)
(out, err) = cmd.communicate()
if not err:
	pod_out=open('%s/cling.pod'%(SCRIPT_DIR), 'w')
	file_handler=open('%s/cling.pod.in'%(SCRIPT_DIR))
	pod_in=file_handler.read()
	print(pod_in.replace("%help_msg%", out.decode()), file=pod_out)
	pod_out.close()
	file_handler.close()
