#!/usr/bin/env python
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
import sys
import threading

from traitlets import Unicode, Float
from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel


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

    def __init__(self, **kwargs):
        super(ClingKernel, self).__init__(**kwargs)
        whichCling = shutil.which('cling')
        if whichCling:
            clingInstDir = os.path.dirname(os.path.dirname(whichCling))
        else:
            #clingInstDir = '/Users/axel/build/cling/cling-all-in-one/clion-inst'
            clingInstDir = '/Users/axel/Library/Caches/CLion12/cmake/generated/e0f22745/e0f22745/Debug'
        self.libclingJupyter = ctypes.CDLL(clingInstDir + "/lib/libclingJupyter.dylib", mode = ctypes.RTLD_GLOBAL)
        self.libclingJupyter.cling_create.restype = ctypes.c_void_p
        self.libclingJupyter.cling_eval.restype = ctypes.c_char_p
        strarr = ctypes.c_char_p*4
        argv = strarr(b"clingJupyter",b"",b"",b"")
        llvmresourcedir = ctypes.c_char_p(clingInstDir.encode('utf8'))
        self.interp = ctypes.c_void_p(self.libclingJupyter.cling_create(4, argv, llvmresourcedir))

        self.libclingJupyter.cling_complete_start.restype = ctypes.c_void_p
        self.libclingJupyter.cling_complete_next.restype = ctypes.c_char_p

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

        if stringResult == 0:
            status = 'error'
        else:
            self.session.send(
                self.iopub_socket,
                'execute_result', 
                content={
                    'data': {
                        'text/plain': stringResult.decode('utf8', replace),
                    },
                },
                parent=self.parent_header
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
                'payload': [{
                  'source': 'set_next_input',
                  'replace': True,
                  'text':'//THIS IS MAGIC\n' + code
                }],
                'user_expressions': {},
            })
        else:
            raise ValueError("Invalid status: %r" % status)
        
        return reply


class ClingKernelApp(IPKernelApp):
    kernel_class = ClingKernel
    def init_io(self):
        # disable io forwarding
        pass


def main():
    """launch a cling kernel"""
    ClingKernelApp.launch_instance()


if __name__ == '__main__':
    main()
