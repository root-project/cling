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
import select
import sys
import threading

try:
    from ipykernel.kernelapp import IPKernelApp
except ImportError:
    from IPython.kernel.zmq.kernelapp import IPKernelApp
try:
    from ipykernel.kernelbase import Kernel
except ImportError:
    from IPython.kernel.zmq.kernelbase import Kernel


libc = ctypes.CDLL(None)
try:
    c_stdout_p = ctypes.c_void_p.in_dll(libc, 'stdout')
except ValueError:
    # libc.stdout is has a funny name on OS X
    c_stdout_p = ctypes.c_void_p.in_dll(libc, '__stdoutp')

try:
    from traitlets import Unicode, Float
except ImportError:
    from IPython.utils.traitlets import Unicode, Float



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
    
    cling = Unicode(config=True,
        help="Path to cling if not on your PATH."
    )
    flush_interval = Float(0.25, config=True)
    
    @contextmanager
    def forward_stdout(self):
        """Capture stdout and forward it as stream messages"""
        # create pipe for stdout
        stdout_fd = sys.__stdout__.fileno()
        save_stdout = os.dup(stdout_fd)
        pipe_out, pipe_in = os.pipe()
        os.dup2(pipe_in, stdout_fd)
        os.close(pipe_in)
        
        # make pipe_out non-blocking
        flags = fcntl(pipe_out, F_GETFL)
        fcntl(pipe_out, F_SETFL, flags|os.O_NONBLOCK)

        def forwarder(pipe, name='stdout'):
            """Forward bytes on a pipe to stream messages"""
            while True:
                r, w, x = select.select([pipe], [], [], self.flush_interval)
                if not r:
                    # nothing to read, flush libc's stdout and check again
                    libc.fflush(c_stdout_p)
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
            libc.fflush(c_stdout_p)
            os.close(stdout_fd)
            t.join()
            
            # and restore original stdout
            os.close(pipe_out)
            os.dup2(save_stdout, stdout_fd)
            os.close(save_stdout)

    def run_cell(self, code, silent=False):
        """Dummy run cell while waiting for cling ctypes API"""
        import time
        
        for i in range(5):
            libc.printf((code + '\n').encode('utf8'))
            time.sleep(0.2 * i)
    
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
        
        with self.forward_stdout():
            self.run_cell(code, silent)

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
                'payload': [],
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
