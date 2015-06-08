#!/usr/bin/env python

from __future__ import print_function

import os
from pipes import quote
import re
import signal
import sys

from tornado.ioloop import IOLoop

from ipykernel.kernelbase import Kernel
from pexpect import replwrap, EOF

__version__ = '0.0.1'

from traitlets import Unicode

class ClingError(Exception):
    def __init__(self, buf):
        self.buf = buf

class ClingInterpreter(replwrap.REPLWrapper):
    
    prompt_pat = re.compile(r'\[cling\][\$\!\?]\s+')
    
    def __init__(self, cmd, **kw):
        self.buffer = []
        self.output = ''
        
        super(ClingInterpreter, self).__init__(
            cmd, '[cling]', None, **kw)
    
    def run_command(self, command, timeout=-1):
        self.buffer = []
        self.output = ''
        try:
            super(ClingInterpreter, self).run_command(command, timeout)
        finally:
            self.output = ''.join(self._squash_raw_input(self.buffer))
            self.buffer = []
        return self.output
    
    def _squash_raw_input(self, buf):
        """kill raw input toggle output"""
        in_raw = False
        for line in buf:
            if in_raw:
                if line.strip() == 'Not using raw input':
                    in_raw = False
            elif line.strip() == 'Using raw input':
                in_raw = True
            else:
                yield line
    
    def _no_echo(self, buf):
        """Filter out cling's input-echo"""
        lines = [ line for line in buf.splitlines(True) if '\x1b[D' not in line ]
        return ''.join(lines)

    def _expect_prompt(self, timeout=-1):
        try:
            self.child.expect(self.prompt_pat, timeout)
        finally:
            if self.child.match and self.child.match.group() == '[cling]! ' and self.child.buffer.startswith('? '):
                self.child.buffer = self.child.buffer[2:].lstrip()
            
            buf = self._no_echo(self.child.before)
            
            if '\x1b[0m\x1b[0;1;31merror:' in buf:
                raise ClingError(buf)
            elif buf:
                self.buffer.append(buf)


class ClingKernel(Kernel):
    implementation = 'cling_kernel'
    implementation_version = __version__

    banner = Unicode()
    def _banner_default(self):
        return 'cling-%s' % self.language_version
        return self._banner

    language_info = {'name': 'c++',
                     'codemirror_mode': 'clike',
                     'mimetype': ' text/x-c++src',
                     'file_extension': '.c++'}
    
    cling = Unicode(config=True,
        help="Path to cling if not on your PATH."
    )
    def _cling_default(self):
        return os.environ.get('CLING_EXE') or 'cling'
    
    def __init__(self, **kwargs):
        
        Kernel.__init__(self, **kwargs)
        self._start_interpreter()

    def _start_interpreter(self):
        # Signal handlers are inherited by forked processes, and we can't easily
        # reset it from the subprocess. Since kernelapp ignores SIGINT except in
        # message handlers, we need to temporarily reset the SIGINT handler here
        # so that bash and its children are interruptible.
        sig = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            self.interpreter = ClingInterpreter(
                '%s --nologo --version' % quote(self.cling)
            )
        finally:
            signal.signal(signal.SIGINT, sig)
        self.language_version = self.interpreter.child.before.strip()

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
        traceback = None
        try:
            # if not clingy:
            #     self.interpreter.run_command('.rawInput')
            output = self.interpreter.run_command(code, timeout=None)
            # if not clingy:
            #     self.interpreter.run_command('.rawInput')
        except KeyboardInterrupt:
            self.interpreter.child.sendintr()
            status = 'interrupted'
            self.interpreter._expect_prompt()
            output = self.interpreter.output
        except EOF:
            # output = self.interpreter._filter_buf(self.interpr)
            output = self.interpreter.output + ' Restarting Cling'
            self._start_interpreter()
        except EOF:
            status = 'error'
            traceback = []
        except ClingError as e:
            status = 'error'
            traceback = e.buf.splitlines()
            output = self.interpreter.output
        if not self.interpreter.child.isalive():
            self.log.error("Cling interpreter died")
            loop = IOLoop.current()
            loop.add_callback(loop.stop)
        
        # print('out: %r' % output, file=sys.__stderr__)
        # print('tb: %r' % traceback, file=sys.__stderr__)
        
        if not silent:
            # Send output on stdout
            stream_content = {'name': 'stdout', 'text': output}
            self.send_response(self.iopub_socket, 'stream', stream_content)

        reply = {
            'status': status,
            'execution_count': self.execution_count,
        }

        if status == 'interrupted':
            pass
        elif status == 'error':
            err = {
                'ename': 'ename',
                'evalue': 'evalue',
                'traceback': traceback,
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

def main():
    """launch a cling kernel"""
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=ClingKernel)


if __name__ == '__main__':
    main()
