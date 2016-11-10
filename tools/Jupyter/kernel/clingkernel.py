#!/usr/bin/env python
#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
# author:  Min RK
# Copyright (c) Min RK
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

"""
Cling Kernel for Jupyter

Talks to Cling via ctypes
"""

from __future__ import print_function

__version__ = '0.0.2'

import ctypes
from contextlib import contextmanager
from fcntl import fcntl, F_GETFL, F_SETFL
import os
import shutil
import select
import struct
import sys
import threading

from traitlets import Unicode, Float, Dict, List, CaselessStrEnum
from ipykernel.kernelbase import Kernel
from ipykernel.kernelapp import kernel_aliases,kernel_flags, IPKernelApp
from ipykernel.ipkernel import IPythonKernel
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython.core.profiledir import ProfileDir
from jupyter_client.session import Session


class my_void_p(ctypes.c_void_p):
  pass

libc = ctypes.CDLL(None)
try:
    c_stdout_p = ctypes.c_void_p.in_dll(libc, 'stdout')
    c_stderr_p = ctypes.c_void_p.in_dll(libc, 'stderr')
except ValueError:
    # libc.stdout is has a funny name on OS X
    c_stdout_p = ctypes.c_void_p.in_dll(libc, '__stdoutp')
    c_stderr_p = ctypes.c_void_p.in_dll(libc, '__stderrp')


class ClingKernel(Kernel):
    """Cling Kernel for Jupyter"""
    implementation = 'cling_kernel'
    implementation_version = __version__
    language_version = 'X'

    banner = Unicode()
    def _banner_default(self):
        return 'cling-%s' % self.language_version
        return self._banner

    # codemirror_mode='clike' *should* work but doesn't, using the mimetype instead
    language_info = {'name': 'c++',
                     'codemirror_mode': 'text/x-c++src',
                     'mimetype': ' text/x-c++src',
                     'file_extension': '.c++'}

    flush_interval = Float(0.25, config=True)

    std = CaselessStrEnum(default_value='c++11',
            values = ['c++11', 'c++14', 'c++17'],
            help="C++ standard to use, either c++17, c++14 or c++11").tag(config=True);

    def __init__(self, **kwargs):
        super(ClingKernel, self).__init__(**kwargs)
        try:
            whichCling = os.readlink(shutil.which('cling'))
        except OSError as e:
            #If cling is not a symlink try a regular file
            #readlink returns POSIX error EINVAL (22) if the
            #argument is not a symlink
            if e.args[0] == 22:
                whichCling = shutil.which('cling')
            else:
                raise e
        except AttributeError:
            from distutils.spawn import find_executable
            whichCling = find_executable('cling')

        if whichCling:
            clingInstDir = os.path.dirname(os.path.dirname(whichCling))
            llvmResourceDir = clingInstDir
        else:
            raise RuntimeError('Cannot find cling in $PATH. No cling, no fun.')

        for ext in ['so', 'dylib', 'dll']:
            libFilename = clingInstDir + "/lib/libclingJupyter." + ext
            if os.access(libFilename, os.R_OK):
                self.libclingJupyter = ctypes.CDLL(clingInstDir + "/lib/libclingJupyter." + ext,
                                                   mode = ctypes.RTLD_GLOBAL)
                break

        if not getattr(self, 'libclingJupyter', None):
            raise RuntimeError('Cannot find ' + clingInstDir + '/lib/libclingJupyter.{so,dylib,dll}')

        self.libclingJupyter.cling_create.restype = my_void_p
        self.libclingJupyter.cling_eval.restype = my_void_p
        #build -std=c++11 or -std=c++14 option
        stdopt = ("-std=" + self.std).encode('utf-8')
        self.log.info("Using {}".format(stdopt.decode('utf-8')))
        #from IPython.utils import io
        #io.rprint("DBG: Using {}".format(stdopt.decode('utf-8')))
        strarr = ctypes.c_char_p*5
        argv = strarr(b"clingJupyter",stdopt, b"-I" + clingInstDir.encode('utf-8') + b"/include/",b"",b"")
        llvmResourceDirCP = ctypes.c_char_p(llvmResourceDir.encode('utf8'))
        self.output_pipe, pipe_in = os.pipe()
        self.interp = self.libclingJupyter.cling_create(5, argv, llvmResourceDirCP, pipe_in)

        self.libclingJupyter.cling_complete_start.restype = my_void_p
        self.libclingJupyter.cling_complete_next.restype = my_void_p #c_char_p
        self.output_thread = threading.Thread(target=self.publish_pipe_output)
        self.output_thread.daemon = True
        self.output_thread.start()

    def _recv_dict(self, pipe):
        """Receive a serialized dict on a pipe

        Returns the dictionary.
        """
        # Wire format:
        #   // Pipe sees (all numbers are longs, except for the first):
        #   // - num bytes in a long (sent as a single unsigned char!)
        #   // - num elements of the MIME dictionary; Jupyter selects one to display.
        #   // For each MIME dictionary element:
        #   //   - length of MIME type key
        #   //   - MIME type key
        #   //   - size of MIME data buffer (including the terminating 0 for
        #   //     0-terminated strings)
        #   //   - MIME data buffer
        data = {}
        b1 = os.read(pipe, 1)
        sizeof_long = struct.unpack('B', b1)[0]
        if sizeof_long == 8:
            fmt = 'Q'
        else:
            fmt = 'L'
        buf = os.read(pipe, sizeof_long)
        num_elements = struct.unpack(fmt, buf)[0]
        for i in range(num_elements):
            buf = os.read(pipe, sizeof_long)
            len_key = struct.unpack(fmt, buf)[0]
            key = os.read(pipe, len_key).decode('utf8')
            buf = os.read(pipe, sizeof_long)
            len_value = struct.unpack(fmt, buf)[0]
            value = os.read(pipe, len_value).decode('utf8')
            data[key] = value
        return data

    def publish_pipe_output(self):
        """Watch output_pipe for display-data messages

        and publish them on IOPub when they arrive
        """

        while True:
            select.select([self.output_pipe], [], [])
            data = self._recv_dict(self.output_pipe)
            self.session.send(self.iopub_socket, 'display_data',
                content={
                    'data': data,
                    'metadata': {},
                },
                parent=self._parent_header,
            )

    @contextmanager
    def forward_stream(self, name):
        """Capture stdout and forward it as stream messages"""
        # create pipe for stdout
        if name == 'stdout':
            c_flush_p = c_stdout_p
        elif name == 'stderr':
            c_flush_p = c_stderr_p
        else:
            raise ValueError("Name must be stdout or stderr, not %r" % name)

        real_fd = getattr(sys, '__%s__' % name).fileno()
        save_fd = os.dup(real_fd)
        pipe_out, pipe_in = os.pipe()
        os.dup2(pipe_in, real_fd)
        os.close(pipe_in)

        # make pipe_out non-blocking
        flags = fcntl(pipe_out, F_GETFL)
        fcntl(pipe_out, F_SETFL, flags|os.O_NONBLOCK)

        def forwarder(pipe):
            """Forward bytes on a pipe to stream messages"""
            while True:
                r, w, x = select.select([pipe], [], [], self.flush_interval)
                if not r:
                    # nothing to read, flush libc's stdout and check again
                    libc.fflush(c_flush_p)
                    continue
                data = os.read(pipe, 1024)
                if not data:
                    # pipe closed, we are done
                    break
                # send output
                self.session.send(self.iopub_socket, 'stream', {
                    'name': name,
                    'text': data.decode('utf8', 'replace'),
                }, parent=self._parent_header)

        t = threading.Thread(target=forwarder, args=(pipe_out,))
        t.start()
        try:
            yield
        finally:
            # flush the pipe
            libc.fflush(c_flush_p)
            os.close(real_fd)
            t.join()

            # and restore original stdout
            os.close(pipe_out)
            os.dup2(save_fd, real_fd)
            os.close(save_fd)

    def run_cell(self, code, silent=False):
        return self.libclingJupyter.cling_eval(self.interp, ctypes.c_char_p(code.encode('utf8')))

    def do_execute(self, code, silent, store_history=True,
                   user_expressions=None, allow_stdin=False):
        if not code.strip():
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }
        status = 'ok'

        with self.forward_stream('stdout'), self.forward_stream('stderr'):
            stringResult = self.run_cell(code, silent)

        if not stringResult:
            status = 'error'
        else:
            self.session.send(
                self.iopub_socket,
                'execute_result',
                content={
                    'data': {
                        'text/plain': ctypes.cast(stringResult, ctypes.c_char_p).value.decode('utf8', 'replace'),
                    },
                    'metadata': {},
                    'execution_count': self.execution_count,
                },
                parent=self._parent_header
            )
            self.libclingJupyter.cling_eval_free(stringResult)


        reply = {
            'status': status,
            'execution_count': self.execution_count,
        }

        if status == 'error':
            err = {
                'ename': 'ename',
                'evalue': 'evalue',
                'traceback': [],
            }
            self.send_response(self.iopub_socket, 'error', err)
            reply.update(err)
        elif status == 'ok':
            reply.update({
                'THIS DOES NOT WORK: payload': [{
                  'source': 'set_next_input',
                  'replace': True,
                  'text':'//THIS IS MAGIC\n' + code
                }],
                'user_expressions': {},
            })
        else:
            raise ValueError("Invalid status: %r" % status)

        return reply

    def do_complete(self, code, cursor_pos):
        """Provide completions here"""
        # if cursor_pos = cursor_start = cursor_end,
        # matches should be a list of strings to be appended after the cursor
        return {'matches' : [],
                'cursor_end' : cursor_pos,
                'cursor_start' : cursor_pos,
                'metadata' : {},
                'status' : 'ok'}

cling_flags = kernel_flags
class ClingKernelApp(IPKernelApp):
    name='cling-kernel'
    cling_aliases = kernel_aliases.copy()
    cling_aliases['std']='ClingKernel.std'
    aliases = Dict(cling_aliases)
    flags = Dict(cling_flags)
    classes = List([ ClingKernel, IPythonKernel, ZMQInteractiveShell, ProfileDir, Session ])
    kernel_class = ClingKernel
    
def main():
    """launch a cling kernel"""
    ClingKernelApp.launch_instance()


if __name__ == '__main__':
    main()
