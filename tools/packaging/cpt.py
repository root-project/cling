#! /usr/bin/env python
# coding:utf-8

###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/cpt.py: Python script to launch Cling Packaging Tool (CPT)
#
# Documentation: tools/packaging/README.md
#
# Author: Anirudha Bose <ani07nov@gmail.com>
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#
###############################################################################

# Python 2 and Python 3 compatibility
from __future__ import print_function

import sys
if sys.version_info < (3,0):
    # Python 2.x
    from urllib2 import urlopen
    input = raw_input
else:
    # Python 3.x
    from urllib.request import urlopen

import argparse
import os
import platform
import subprocess
import shutil
import shlex
import glob
import re
import tarfile
import zipfile
from email.utils import formatdate
from datetime import tzinfo
import time
import multiprocessing
import fileinput
import stat

###############################################################################
#              Platform independent functions (formerly indep.py)             #
###############################################################################

def _convert_subprocess_cmd(cmd):
    if OS == 'Windows':
        if '"' in cmd:
            # Assume there's only one quoted argument.
            bits = cmd.split('"')
            return bits[0].split() + [bits[1]] + bits[2].split()
        else:
            return cmd.split()
    else:
        return [cmd]

def _perror(e):
    print("subprocess.CalledProcessError: Command '%s' returned non-zero exit status %s"%(' '.join(e.cmd), str(e.returncode)))
    cleanup()
    # Communicate return code to the calling program if any
    sys.exit(e.returncode)

def exec_subprocess_call(cmd, cwd):
    cmd = _convert_subprocess_cmd(cmd)
    try:
        subprocess.check_call(cmd, cwd=cwd, shell=True,
                              stdin=subprocess.PIPE, stdout=None, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        _perror(e)

def exec_subprocess_check_output(cmd, cwd):
    cmd = _convert_subprocess_cmd(cmd)
    try:
        out = subprocess.check_output(cmd, cwd=cwd, shell=True,
                                      stdin=subprocess.PIPE, stderr=subprocess.STDOUT).decode('utf-8')
    except subprocess.CalledProcessError as e:
        _perror(e)

    finally:
        return out

def box_draw_header():
    msg='cling (' + platform.machine() + ')' + formatdate(time.time(),tzinfo())
    spaces_no = 80 - len(msg) - 4
    spacer = ' ' * spaces_no
    msg='cling (' + platform.machine() + ')' + spacer + formatdate(time.time(),tzinfo())

    if OS != 'Windows':
        print('''
╔══════════════════════════════════════════════════════════════════════════════╗
║ %s ║
╚══════════════════════════════════════════════════════════════════════════════╝'''%(msg))
    else:
        print('''
+=============================================================================+
| %s|
+=============================================================================+'''%(msg))


def box_draw(msg):
    spaces_no = 80 - len(msg) - 4
    spacer = ' ' * spaces_no

    if OS != 'Windows':
        print('''
┌──────────────────────────────────────────────────────────────────────────────┐
│ %s%s │
└──────────────────────────────────────────────────────────────────────────────┘'''%(msg, spacer))
    else:
        print('''
+-----------------------------------------------------------------------------+
| %s%s|
+-----------------------------------------------------------------------------+'''%(msg, spacer))

def wget(url, out_dir):
    file_name = url.split('/')[-1]
    print("HTTP request sent, awaiting response... ")
    u = urlopen(url)
    if u.code == 200:
        print("Connected to %s [200 OK]"%(url))
    else:
        exit()
    f = open(os.path.join(out_dir, file_name), 'wb')
    if sys.version_info < (3,0):
        # Python 2.x
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
    else:
        # Python 3.x
        file_size = int(u.getheader("Content-Length"))

    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status, end=' ')
    f.close()

def fetch_llvm():
    box_draw("Fetch source files")
    print('Last known good LLVM revision is: ' + LLVMRevision)
    print('Current working directory is: ' + workdir + '\n')
    def get_fresh_llvm():
        exec_subprocess_call('git clone %s %s'%(LLVM_GIT_URL, srcdir), workdir)

        exec_subprocess_call('git checkout cling-patches-r%s'%(LLVMRevision), srcdir)

    def update_old_llvm():
        exec_subprocess_call('git stash', srcdir)

        exec_subprocess_call('git clean -f -x -d', srcdir)

        exec_subprocess_call('git fetch --tags', srcdir)

        exec_subprocess_call('git checkout cling-patches-r%s'%(LLVMRevision), srcdir)

        exec_subprocess_call('git pull origin refs/tags/cling-patches-r%s'%(LLVMRevision), srcdir)

    if os.path.isdir(srcdir):
        update_old_llvm()
    else:
        get_fresh_llvm()


def fetch_clang():
    def get_fresh_clang():
        exec_subprocess_call('git clone %s'%(CLANG_GIT_URL), os.path.join(srcdir, 'tools'))

        exec_subprocess_call('git checkout cling-patches-r%s'%(LLVMRevision), os.path.join(srcdir, 'tools', 'clang'))

    def update_old_clang():
        exec_subprocess_call('git stash', os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git clean -f -x -d', os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git fetch --tags', os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git checkout cling-patches-r%s'%(LLVMRevision), os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git pull origin refs/tags/cling-patches-r%s'%(LLVMRevision), os.path.join(srcdir, 'tools', 'clang'))

    if os.path.isdir(os.path.join(srcdir, 'tools', 'clang')):
        update_old_clang()
    else:
        get_fresh_clang()

def fetch_cling(arg):
    def get_fresh_cling():
        exec_subprocess_call('git clone %s'%(CLING_GIT_URL), os.path.join(srcdir, 'tools'))

        if arg == 'last-stable':
            checkout_branch = exec_subprocess_check_output('git describe --match v* --abbrev=0 --tags | head -n 1', CLING_SRC_DIR)

        elif arg == 'master':
            checkout_branch = 'master'
        else:
            checkout_branch = arg

        exec_subprocess_call('git checkout %s'%(checkout_branch), CLING_SRC_DIR)

    def update_old_cling():
        exec_subprocess_call('git stash', CLING_SRC_DIR)

        exec_subprocess_call('git clean -f -x -d', CLING_SRC_DIR)

        exec_subprocess_call('git fetch --tags', CLING_SRC_DIR)

        if arg == 'last-stable':
            checkout_branch = exec_subprocess_check_output('git describe --match v* --abbrev=0 --tags | head -n 1', CLING_SRC_DIR)

        elif arg == 'master':
            checkout_branch = 'master'
        else:
            checkout_branch = arg

        exec_subprocess_call('git checkout %s'%(checkout_branch), CLING_SRC_DIR)

        exec_subprocess_call('git pull origin %s'%(checkout_branch), CLING_SRC_DIR)

    if os.path.isdir(CLING_SRC_DIR):
        update_old_cling()
    else:
        get_fresh_cling()


def set_version():
    global VERSION
    global REVISION
    box_draw("Set Cling version")
    VERSION=open(os.path.join(CLING_SRC_DIR, 'VERSION'), 'r').readline().strip()

    # If development release, then add revision to the version
    REVISION = exec_subprocess_check_output('git log -n 1 --pretty=format:%H', CLING_SRC_DIR).strip()

    if '~dev' in VERSION:
        VERSION = VERSION + '-' + REVISION[:7]

    print('Version: ' + VERSION)
    print('Revision: ' + REVISION)


def set_vars():
    global EXEEXT
    global SHLIBEXT
    global CLANG_VERSION
    box_draw("Set variables")
    if not os.path.isfile(os.path.join(LLVM_OBJ_ROOT, 'test', 'lit.site.cfg')):
        exec_subprocess_call('make lit.site.cfg', os.path.join(LLVM_OBJ_ROOT, 'test'))

    with open(os.path.join(LLVM_OBJ_ROOT, 'test', 'lit.site.cfg'), 'r') as lit_site_cfg:
        for line in lit_site_cfg:
            if re.match('^config.llvm_shlib_ext = ', line):
                SHLIBEXT = re.sub('^config.llvm_shlib_ext = ', '', line).replace('"', '').strip()
            elif re.match('^config.llvm_exe_ext = ', line):
                EXEEXT = re.sub('^config.llvm_exe_ext = ', '', line).replace('"', '').strip()

    if not os.path.isfile(os.path.join(LLVM_OBJ_ROOT, 'tools', 'clang', 'include', 'clang', 'Basic', 'Version.inc')):
        exec_subprocess_call('make Version.inc', os.path.join(LLVM_OBJ_ROOT, 'tools', 'clang', 'include', 'clang', 'Basic'))

    with open(os.path.join(LLVM_OBJ_ROOT, 'tools', 'clang', 'include', 'clang', 'Basic', 'Version.inc'), 'r') as Version_inc:
        for line in Version_inc:
            if re.match('^#define CLANG_VERSION ', line):
                CLANG_VERSION = re.sub('^#define CLANG_VERSION ', '', line).strip()

    print('EXEEXT: ' + EXEEXT)
    print('SHLIBEXT: ' + SHLIBEXT)
    print('CLANG_VERSION: ' + CLANG_VERSION)

def compile(arg):
    global prefix
    prefix=arg
    PYTHON=sys.executable
    cores=multiprocessing.cpu_count()

    # Cleanup previous installation directory if any
    if os.path.isdir(prefix):
        print("Remove directory: " + prefix)
        shutil.rmtree(prefix)

    # Cleanup previous build directory if exists
    if os.path.isdir(os.path.join(workdir, 'builddir')):
        print("Remove directory: " + os.path.join(workdir, 'builddir'))
        shutil.rmtree(os.path.join(workdir, 'builddir'))

    os.makedirs(os.path.join(workdir, 'builddir'))

    if platform.system() == 'Windows':
        CMAKE = os.path.join(TMP_PREFIX, 'bin', 'cmake', 'bin', 'cmake.exe')

        if args['create_dev_env'] == 'debug':
            box_draw("Configure Cling with CMake and generate Visual Studio 11 project files")
            exec_subprocess_call('%s -G "Visual Studio 11" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=%s ..\%s'%(CMAKE, TMP_PREFIX, os.path.basename(srcdir)), LLVM_OBJ_ROOT)

            box_draw("Building Cling (using %s cores)"%(cores))
            exec_subprocess_call('%s --build . --target clang --config Debug'%(CMAKE), LLVM_OBJ_ROOT)

            exec_subprocess_call('%s --build . --target cling --config Debug'%(CMAKE), LLVM_OBJ_ROOT)

        else:
            box_draw("Configure Cling with CMake and generate Visual Studio 11 project files")
            exec_subprocess_call('%s -G "Visual Studio 11" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%s ..\%s'%(CMAKE, TMP_PREFIX, os.path.basename(srcdir)), LLVM_OBJ_ROOT)

            box_draw("Building Cling (using %s cores)"%(cores))
            exec_subprocess_call('%s --build . --target clang --config Release'%(CMAKE), LLVM_OBJ_ROOT)

            exec_subprocess_call('%s --build . --target cling --config Release'%(CMAKE), LLVM_OBJ_ROOT)

            box_draw("Install compiled binaries to prefix (using %s cores)"%(cores))
            exec_subprocess_call('%s --build . --target INSTALL --config Release'%(CMAKE), LLVM_OBJ_ROOT)

    else:
        box_draw("Configure Cling with GNU Make")

        if args['create_dev_env'] == 'debug':
            exec_subprocess_call('%s/configure --disable-compiler-version-checks --with-python=%s --enable-targets=host --prefix=%s --enable-cxx11'%(srcdir, PYTHON, TMP_PREFIX), LLVM_OBJ_ROOT)

        else:
            exec_subprocess_call('%s/configure --disable-compiler-version-checks --with-python=%s --enable-targets=host --prefix=%s --enable-optimized=yes --enable-cxx11'%(srcdir, PYTHON, TMP_PREFIX), LLVM_OBJ_ROOT)

        box_draw("Building Cling (using %s cores)"%(cores))
        exec_subprocess_call('make -j%s'%(cores), LLVM_OBJ_ROOT)

        box_draw("Install compiled binaries to prefix (using %s cores)"%(cores))
        exec_subprocess_call('make install -j%s prefix=%s'%(cores, TMP_PREFIX), LLVM_OBJ_ROOT)


def install_prefix():
    set_vars()
    box_draw("Filtering Cling's libraries and binaries")

    for line in fileinput.input(os.path.join(CLING_SRC_DIR, 'tools', 'packaging', 'dist-files.mk'), inplace=True):
        if '@EXEEXT@' in line:
            print(line.replace('@EXEEXT@', EXEEXT), end=' ')
        elif '@SHLIBEXT@' in line:
            print(line.replace('@SHLIBEXT@', SHLIBEXT), end=' ')
        elif '@CLANG_VERSION@' in line:
            print(line.replace('@CLANG_VERSION@', CLANG_VERSION), end=' ')
        else:
            print(line, end=' ')

    dist_files = open(os.path.join(CLING_SRC_DIR, 'tools', 'packaging', 'dist-files.mk'), 'r').read()
    for root, dirs, files in os.walk(TMP_PREFIX):
        for file in files:
            f=os.path.join(root, file).replace(TMP_PREFIX, '')
            if f.lstrip(os.sep).replace(os.sep, '/')+' ' in dist_files:
                print("Filter: " + f)
                if not os.path.isdir(os.path.join(prefix,os.path.dirname(f))):
                    os.makedirs(os.path.join(prefix,os.path.dirname(f)))
                shutil.copy(os.path.join(TMP_PREFIX,f), os.path.join(prefix,f))

def test_cling():
    box_draw("Run Cling test suite")
    if platform.system() != 'Windows':
        exec_subprocess_call('make test', os.path.join(workdir, 'builddir', 'tools', 'cling'))


def tarball():
    box_draw("Compress binaries into a bzip2 tarball")
    tar = tarfile.open(prefix+'.tar.bz2', 'w:bz2')
    print('Creating archive: ' + os.path.basename(prefix) + '.tar.bz2')
    tar.add(prefix, arcname=os.path.basename(prefix))
    tar.close()


def cleanup():
    print("\n")
    box_draw("Clean up")
    if os.path.isdir(os.path.join(workdir, 'builddir')):
        print("Remove directory: " + os.path.join(workdir, 'builddir'))
        shutil.rmtree(os.path.join(workdir, 'builddir'))

    if os.path.isdir(prefix):
        print("Remove directory: " + prefix)
        shutil.rmtree(prefix)

    if os.path.isdir(TMP_PREFIX):
        print("Remove directory: " + TMP_PREFIX)
        shutil.rmtree(TMP_PREFIX)

    if os.path.isfile(os.path.join(workdir,'cling.nsi')):
        print("Remove file: " + os.path.join(workdir,'cling.nsi'))
        os.remove(os.path.join(workdir,'cling.nsi'))

    if args['current_dev'] == 'deb' or args['last_stable'] == 'deb' or args['deb_tag']:
        print('Create output directory: ' + os.path.join(workdir, 'cling-%s-1'%(VERSION)))
        os.makedirs(os.path.join(workdir, 'cling-%s-1'%(VERSION)))

        for file in glob.glob(os.path.join(workdir, 'cling_%s*'%(VERSION))):
            print(file + '->' + os.path.join(workdir, 'cling-%s-1'%(VERSION), os.path.basename(file)))
            shutil.move(file, os.path.join(workdir, 'cling-%s-1'%(VERSION)))

        if not os.listdir(os.path.join(workdir, 'cling-%s-1'%(VERSION))):
            os.rmdir(os.path.join(workdir, 'cling-%s-1'%(VERSION)))

    if args['current_dev'] == 'dmg' or args['last_stable'] == 'dmg' or args['dmg_tag']:
        if os.path.isfile(os.path.join(workdir,'cling-%s-temp.dmg'%(VERSION))):
            print("Remove file: " + os.path.join(workdir,'cling-%s-temp.dmg'%(VERSION)))
            os.remove(os.path.join(workdir,'cling-%s-temp.dmg'%(VERSION)))

        if os.path.isdir(os.path.join(workdir, 'Cling.app')):
            print('Remove directory: ' + 'Cling.app')
            shutil.rmtree(os.path.join(workdir, 'Cling.app'))

        if os.path.isdir(os.path.join(workdir,'cling-%s-temp.dmg'%(VERSION))):
            print('Remove directory: ' + os.path.join(workdir,'cling-%s-temp.dmg'%(VERSION)))
            shutil.rmtree(os.path.join(workdir,'cling-%s-temp.dmg'%(VERSION)))

        if os.path.isdir(os.path.join(workdir, 'Install')):
            print('Remove directory: ' + os.path.join(workdir, 'Install'))
            shutil.rmtree(os.path.join(workdir, 'Install'))

###############################################################################
#            Debian specific functions (ported from debianize.sh)             #
###############################################################################

def check_ubuntu(pkg):
    if pkg == "gnupg":
        SIGNING_USER = exec_subprocess_check_output('gpg --fingerprint | grep uid | sed s/"uid *"//g', '/').strip()
        if SIGNING_USER == '':
            print(pkg.ljust(20) + '[INSTALLED - NOT SETUP]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))
    elif pkg == "python":
        if float(platform.python_version()[:3]) < 2.7:
            print(pkg.ljust(20) + '[OUTDATED VERSION (<2.7)]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))
    elif pkg == "SSL":
        import socket
        if hasattr(socket, 'ssl'):
            print(pkg.ljust(20) + '[SUPPORTED]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[NOT SUPPORTED]'.ljust(30))
    elif exec_subprocess_check_output("dpkg-query -W -f='${Status}' %s 2>/dev/null | grep -c 'ok installed'"%(pkg), '/').strip() == '0':
        print(pkg.ljust(20) + '[NOT INSTALLED]'.ljust(30))
    else:
        if pkg == "gcc":
            if float(exec_subprocess_check_output('gcc -dumpversion', '/')[:3].strip()) <= 4.7:
                print(pkg.ljust(20) + '[UNSUPPORTED VERSION (<4.7)]'.ljust(30))
            else:
                print(pkg.ljust(20) + '[OK]'.ljust(30))
        elif pkg == "g++":
            if float(exec_subprocess_check_output('g++ -dumpversion', '/')[:3].strip()) <= 4.7:
                print(pkg.ljust(20) + '[UNSUPPORTED VERSION (<4.7)]'.ljust(30))
            else:
                print(pkg.ljust(20) + '[OK]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))


def tarball_deb():
    box_draw("Compress compiled binaries into a bzip2 tarball")
    tar = tarfile.open(os.path.join(workdir, 'cling_' + VERSION +'.orig.tar.bz2'), 'w:bz2')
    tar.add(prefix, arcname=os.path.basename(prefix))
    tar.close()

def debianize():
    SIGNING_USER = exec_subprocess_check_output('gpg --fingerprint | grep uid | sed s/"uid *"//g', CLING_SRC_DIR).strip()

    box_draw("Set up the debian directory")
    print("Create directory: debian")
    os.makedirs(os.path.join(prefix, 'debian'))

    print("Create directory: " + os.path.join(prefix, 'debian', 'source'))
    os.makedirs(os.path.join(prefix, 'debian', 'source'))

    print("Create file: " + os.path.join(prefix, 'debian', 'source', 'format'))
    f=open(os.path.join(prefix, 'debian', 'source', 'format'), 'w')
    f.write('3.0 (quilt)')
    f.close()

    print("Create file: " + os.path.join(prefix, 'debian', 'source', 'lintian-overrides'))
    f=open(os.path.join(prefix, 'debian', 'source', 'lintian-overrides'), 'w')
    f.write('cling source: source-is-missing')
    f.close()

    # This section is no longer valid. I have kept it as a reference if we plan to
    # distribute libcling.so or any other library with the package.
    if False:
        print('Create file: ' + os.path.join(prefix, 'debian', 'postinst'))
        template = '''
#! /bin/sh -e
# postinst script for cling
#
# see: dh_installdeb(1)

set -e

# Call ldconfig on libclang.so
ldconfig -l /usr/lib/libclang.so

# dh_installdeb will replace this with shell code automatically
# generated by other debhelper scripts.

#DEBHELPER#

exit 0
'''
        f=open(os.path.join(prefix, 'debian', 'postinst'), 'w')
        f.write(template)
        f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'cling.install'))
    f=open(os.path.join(prefix, 'debian', 'cling.install'), 'w')
    template ='''
bin/* /usr/bin
docs/* /usr/share/doc
include/* /usr/include
lib/* /usr/lib
share/* /usr/share
'''
    f.write(template.strip())
    f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'compact'))
    # Optimize binary compression
    f = open(os.path.join(prefix, 'debian', 'compact'), 'w')
    f.write("7")
    f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'compat'))
    f = open(os.path.join(prefix, 'debian', 'compat'), 'w')
    f.write("9")
    f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'control'))
    f = open(os.path.join(prefix, 'debian', 'control'), 'w')
    template = '''
Source: cling
Section: devel
Priority: optional
Maintainer: Cling Developer Team <cling-dev@cern.ch>
Uploaders: %s
Build-Depends: debhelper (>= 9.0.0)
Standards-Version: 3.9.5
Homepage: http://cling.web.cern.ch/
Vcs-Git: http://root.cern.ch/git/cling.git
Vcs-Browser: http://root.cern.ch/gitweb?p=cling.git;a=summary

Package: cling
Priority: optional
Architecture: any
Depends: ${shlibs:Depends}, ${misc:Depends}
Description: interactive C++ interpreter
 Cling is a new and interactive C++11 standard compliant interpreter built
 on the top of Clang and LLVM compiler infrastructure. Its advantages over
 the standard interpreters are that it has command line prompt and uses
 Just In Time (JIT) compiler for compilation. Many of the developers
 (e.g. Mono in their project called CSharpRepl) of such kind of software
 applications name them interactive compilers.
 .
 One of Cling's main goals is to provide contemporary, high-performance
 alternative of the current C++ interpreter in the ROOT project - CINT. Cling
 serves as a core component of the ROOT system for storing and analyzing the
 data of the Large Hadron Collider (LHC) experiments. The
 backward-compatibility with CINT is major priority during the development.
'''%(SIGNING_USER)
    f.write(template.strip())
    f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'copyright'))
    f = open(os.path.join(prefix, 'debian', 'copyright'), 'w')
    template = '''
Format: http://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: cling
Source: http://root.cern.ch/gitweb?p=cling.git;a=summary

Files: *
Copyright: 2007-2014 by the Authors
License: LGPL-2.0+
Comment: Developed by The ROOT Team; CERN and Fermilab

Files: debian/*
Copyright: 2014 Anirudha Bose <ani07nov@gmail.com>
License: LGPL-2.0+

License: LGPL-2.0+
 This package is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2 of the License, or (at your option) any later version.
 .
 This package is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 .
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
 .
 On Debian systems, the complete text of the GNU Lesser General
 Public License can be found in "/usr/share/common-licenses/LGPL-2".
Comment: Cling can also be licensed under University of Illinois/NCSA
 Open Source License (UI/NCSAOSL).
 .
 More information here: http://root.cern.ch/gitweb?p=cling.git;a=blob_plain;f=LICENSE.TXT;hb=HEAD
'''
    f.write(template.strip())
    f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'rules'))
    f = open(os.path.join(prefix, 'debian', 'rules'), 'w')
    template = '''
#!/usr/bin/make -f
# -*- makefile -*-

%:
	dh $@

override_dh_auto_build:

override_dh_auto_install:
'''
    f.write(template.strip())
    f.close()

    print('Create file: ' + os.path.join(prefix, 'debian', 'changelog'))
    f = open(os.path.join(prefix, 'debian', 'changelog'), 'w')

    template = '''
cling (%s-1) unstable; urgency=low

  * [Debian] Upload to unstable for version: %s
'''%(VERSION, VERSION)
    f.write(template.lstrip())
    f.close()

    if '~dev' in VERSION:
        TAG = str(float(VERSION[:VERSION.find('~')]) - 0.1)
        template = exec_subprocess_check_output('git log v' + TAG + '...HEAD --format="  * %s" | fmt -s', CLING_SRC_DIR)

        f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
        f.write(template)
        f.close()

        f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
        f.write('\n -- ' + SIGNING_USER + '  ' + formatdate(time.time(),tzinfo()) + '\n')
        f.close()
    else:
        TAG=VERSION.replace('v', '')
        if TAG == '0.1':
            f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
            f.write('\n -- ' + SIGNING_USER + '  ' + formatdate(time.time(),tzinfo()) + '\n')
            f.close()
        STABLE_FLAG = '1'

    while TAG != '0.1':
        CMP = TAG
        TAG= str(float(TAG) - 0.1)
        if STABLE_FLAG != '1':
            f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
            f.write('cling (' + TAG + '-1) unstable; urgency=low\n')
            f.close()
            STABLE_FLAG = '1'
            template = exec_subprocess_check_output('git log v' + CMP + '...v' + TAG + '--format="  * %s" | fmt -s', CLING_SRC_DIR)

            f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
            f.write(template)
            f.close()

            f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
            f.write('\n -- ' + SIGNING_USER + '  ' + formatdate(time.time(),tzinfo()) + '\n')
            f.close()

    # Changelog entries from first commit to v0.1
    f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
    f.write('\nOld Changelog:\n')
    f.close()

    template = exec_subprocess_check_output('git log v0.1 --format="  * %s%n -- %an <%ae>  %cD%n"', CLING_SRC_DIR)

    f = open(os.path.join(prefix, 'debian', 'changelog'), 'a+')
    f.write(template.encode('utf-8'))
    f.close()

    box_draw("Run debuild to create Debian package")
    exec_subprocess_call('debuild', prefix)

###############################################################################
#                          Red Hat specific functions                         #
###############################################################################

def check_redhat(pkg):
    if pkg == "python":
        if platform.python_version()[0] == '3':
            print(pkg.ljust(20) + '[UNSUPPORTED VERSION (Python 3)]'.ljust(30))
        elif float(platform.python_version()[:3]) < 2.7:
            print(pkg.ljust(20) + '[OUTDATED VERSION (<2.7)]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))
    elif pkg == "SSL":
        import socket
        if hasattr(socket, 'ssl'):
            print(pkg.ljust(20) + '[SUPPORTED]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[NOT SUPPORTED]'.ljust(30))
    elif exec_subprocess_check_output("rpm -qa | grep -w %s"%(pkg), '/').strip() == '':
        print(pkg.ljust(20) + '[NOT INSTALLED]'.ljust(30))
    else:
        if pkg == "gcc-c++":
            if float(exec_subprocess_check_output('g++ -dumpversion', '/')[:3].strip()) <= 4.7:
                print(pkg.ljust(20) + '[UNSUPPORTED VERSION (<4.7)]'.ljust(30))
            else:
                print(pkg.ljust(20) + '[OK]'.ljust(30))
        elif pkg == "gcc":
            if float(exec_subprocess_check_output('gcc -dumpversion', '/')[:3].strip()) <= 4.7:
                print(pkg.ljust(20) + '[UNSUPPORTED VERSION (<4.7)]'.ljust(30))
            else:
                print(pkg.ljust(20) + '[OK]'.ljust(30))

        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))


def rpm_build():
    global REVISION
    box_draw("Set up RPM build environment")
    if os.path.isdir(os.path.join(workdir, 'rpmbuild')):
        shutil.rmtree(os.path.join(workdir, 'rpmbuild'))
    os.makedirs(os.path.join(workdir, 'rpmbuild'))
    os.makedirs(os.path.join(workdir, 'rpmbuild', 'RPMS'))
    os.makedirs(os.path.join(workdir, 'rpmbuild', 'BUILD'))
    os.makedirs(os.path.join(workdir, 'rpmbuild', 'SOURCES'))
    os.makedirs(os.path.join(workdir, 'rpmbuild', 'SPECS'))
    os.makedirs(os.path.join(workdir, 'rpmbuild', 'tmp'))
    shutil.move(os.path.join(workdir, os.path.basename(prefix)+'.tar.bz2'), os.path.join(workdir, 'rpmbuild', 'SOURCES'))


    box_draw("Generate RPM SPEC file")
    print('Create file: ' + os.path.join(workdir, 'rpmbuild', 'SPECS', 'cling-%s.spec'%(VERSION)))
    f = open(os.path.join(workdir, 'rpmbuild', 'SPECS', 'cling-%s.spec'%(VERSION)), 'w')

    if REVISION == '':
        REVISION = '1'

    template = '''
%define        __spec_install_post %{nil}
%define          debug_package %{nil}
%define        __os_install_post %{_dbpath}/brp-compress

Summary: Interactive C++ interpreter
Name: cling
Version: 0.2~dev
Release: ''' + REVISION[:7] + '''
License: LGPLv2+ or NCSA
Group: Development/Languages/Other
SOURCE0 : %{name}-%{version}.tar.bz2
URL: http://cling.web.cern.ch/
Vendor: Developed by The ROOT Team; CERN and Fermilab
Packager: Anirudha Bose <ani07nov@gmail.com>

BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root

%description
Cling is a new and interactive C++11 standard compliant interpreter built
on the top of Clang and LLVM compiler infrastructure. Its advantages over
the standard interpreters are that it has command line prompt and uses
Just In Time (JIT) compiler for compilation. Many of the developers
(e.g. Mono in their project called CSharpRepl) of such kind of software
applications name them interactive compilers.

One of Cling's main goals is to provide contemporary, high-performance
alternative of the current C++ interpreter in the ROOT project - CINT. Cling
serves as a core component of the ROOT system for storing and analyzing the
data of the Large Hadron Collider (LHC) experiments. The
backward-compatibility with CINT is major priority during the development.

%prep
%setup
mkdir -p $RPM_BUILD_DIR/%{name}-%{version}/usr/share/doc
mv $RPM_BUILD_DIR/%{name}-%{version}/bin/ $RPM_BUILD_DIR/%{name}-%{version}/usr
mv $RPM_BUILD_DIR/%{name}-%{version}/docs/* $RPM_BUILD_DIR/%{name}-%{version}/usr/share/doc/
mv $RPM_BUILD_DIR/%{name}-%{version}/lib/ $RPM_BUILD_DIR/%{name}-%{version}/usr
mv $RPM_BUILD_DIR/%{name}-%{version}/include/ $RPM_BUILD_DIR/%{name}-%{version}/usr
mv $RPM_BUILD_DIR/%{name}-%{version}/share/* $RPM_BUILD_DIR/%{name}-%{version}/usr/share

rm -Rf $RPM_BUILD_DIR/%{name}-%{version}/docs
rm -Rf $RPM_BUILD_DIR/%{name}-%{version}/share

if [ ${RPM_ARCH} = 'x86_64' ]; then
    mv $RPM_BUILD_DIR/%{name}-%{version}/usr/lib $RPM_BUILD_DIR/%{name}-%{version}/usr/lib64
fi

%build
# Empty section.

%install
rm -rf %{buildroot}
mkdir -p  %{buildroot}

# in builddir
cp -a * %{buildroot}


%clean
rm -rf %{buildroot}

%files
%defattr(-,root,root,-)
%{_bindir}/*
%{_includedir}/*
%{_libdir}/*
%{_datadir}/*

%changelog
* Sun Apr 13 2014  Anirudha Bose <ani07nov@gmail.com>
- Initial SPEC file of Cling for RPM packaging
'''
    f.write(template.strip())
    f.close()

    box_draw('Run rpmbuild program')
    exec_subprocess_call('rpmbuild --define "_topdir ${PWD}" -bb %s'%(os.path.join(workdir, 'rpmbuild', 'SPECS', 'cling-%s.spec'%(VERSION))), os.path.join(workdir, 'rpmbuild'))

###############################################################################
#           Windows specific functions (ported from windows_dep.sh)           #
###############################################################################

def check_win(pkg):
    # Check for Microsoft Visual Studio 11.0
    if pkg == "msvc":
        if exec_subprocess_check_output('REG QUERY HKEY_CLASSES_ROOT\VisualStudio.DTE.11.0', 'C:\\').find('ERROR') == -1:
            print(pkg.ljust(20) + '[OK]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[NOT INSTALLED]'.ljust(30))

    elif pkg == "python":
        if platform.python_version()[0] == '3':
            print(pkg.ljust(20) + '[UNSUPPORTED VERSION (Python 3)]'.ljust(30))
        elif float(platform.python_version()[:3]) < 2.7:
            print(pkg.ljust(20) + '[OUTDATED VERSION (<2.7)]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))
    elif pkg == 'SSL':
        import socket
        if hasattr(socket, 'ssl'):
            print(pkg.ljust(20) + '[SUPPORTED]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[NOT SUPPORTED]'.ljust(30))

  # Check for other tools
    else:
        if exec_subprocess_check_output('where %s'%(pkg), 'C:\\').find('INFO: Could not find files for the given pattern') != -1:
            print(pkg.ljust(20) + '[NOT INSTALLED]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))

def get_win_dep():
    box_draw("Download NSIS compiler")
    html = urlopen('http://sourceforge.net/p/nsis/code/HEAD/tree/NSIS/tags/').read().decode('utf-8')
    NSIS_VERSION = html[html.rfind('<a href="v'):html.find('>', html.rfind('<a href="v'))].strip('<a href="v').strip('"')
    NSIS_VERSION = NSIS_VERSION[:1] + '.' + NSIS_VERSION[1:]
    print('Latest version of NSIS is: ' + NSIS_VERSION)
    wget(url="http://sourceforge.net/projects/nsis/files/NSIS%%203%%20Pre-release/%s/nsis-%s.zip"%(NSIS_VERSION, NSIS_VERSION),
         out_dir=TMP_PREFIX)
    print('Extracting: ' + os.path.join(TMP_PREFIX, 'nsis-%s.zip'%(NSIS_VERSION)))
    zip = zipfile.ZipFile(os.path.join(TMP_PREFIX, 'nsis-%s.zip'%(NSIS_VERSION)))
    zip.extractall(os.path.join(TMP_PREFIX, 'bin'))
    print('Remove file: ' + os.path.join(TMP_PREFIX, 'nsis-%s.zip'%(NSIS_VERSION)))
    os.rename(os.path.join(TMP_PREFIX, 'bin', 'nsis-%s'%(NSIS_VERSION)), os.path.join(TMP_PREFIX, 'bin', 'nsis'))

    box_draw("Download CMake for Windows")
    html = urlopen('http://www.cmake.org/cmake/resources/software.html').read().decode('utf-8')
    CMAKE_VERSION = html[html.find('Latest Release ('): html.find(')', html.find('Latest Release ('))].strip('Latest Release (')
    print('Latest stable version of CMake is: ' + CMAKE_VERSION)
    wget(url='http://www.cmake.org/files/v%s/cmake-%s-win32-x86.zip'%(CMAKE_VERSION[:3], CMAKE_VERSION),
         out_dir=TMP_PREFIX)
    print('Extracting: ' + os.path.join(TMP_PREFIX, 'cmake-%s-win32-x86.zip'%(CMAKE_VERSION)))
    zip = zipfile.ZipFile(os.path.join(TMP_PREFIX, 'cmake-%s-win32-x86.zip'%(CMAKE_VERSION)))
    zip.extractall(os.path.join(TMP_PREFIX, 'bin'))
    print('Remove file: ' + os.path.join(TMP_PREFIX, 'cmake-%s-win32-x86.zip'%(CMAKE_VERSION)))
    os.rename(os.path.join(TMP_PREFIX, 'bin', 'cmake-%s-win32-x86'%(CMAKE_VERSION)), os.path.join(TMP_PREFIX, 'bin', 'cmake'))

def make_nsi():
    box_draw("Generating cling.nsi")
    NSIS = os.path.join(TMP_PREFIX, 'bin', 'nsis')
    VIProductVersion = exec_subprocess_check_output('git describe --match v* --abbrev=0 --tags', CLING_SRC_DIR).strip().splitlines()[0]
    print('Create file: ' + os.path.join(workdir, 'cling.nsi'))
    f = open(os.path.join(workdir, 'cling.nsi'), 'w')
    template = '''
; Cling setup script %s
!define APP_NAME "Cling"
!define COMP_NAME "CERN"
!define WEB_SITE "http://cling.web.cern.ch/"
!define VERSION "%s"
!define COPYRIGHT "Copyright © 2007-2014 by the Authors; Developed by The ROOT Team, CERN and Fermilab"
!define DESCRIPTION "Interactive C++ interpreter"
!define INSTALLER_NAME "%s"
!define MAIN_APP_EXE "cling.exe"
!define INSTALL_TYPE "SetShellVarContext current"
!define PRODUCT_ROOT_KEY "HKLM"
!define PRODUCT_KEY "Software\Cling"

###############################################################################

VIProductVersion  "%s.0.0"
VIAddVersionKey "ProductName"  "${APP_NAME}"
VIAddVersionKey "CompanyName"  "${COMP_NAME}"
VIAddVersionKey "LegalCopyright"  "${COPYRIGHT}"
VIAddVersionKey "FileDescription"  "${DESCRIPTION}"
VIAddVersionKey "FileVersion"  "${VERSION}"

###############################################################################

SetCompressor /SOLID Lzma
Name "${APP_NAME}"
Caption "${APP_NAME}"
OutFile "${INSTALLER_NAME}"
BrandingText "${APP_NAME}"
XPStyle on
InstallDir "C:\\Cling\\cling-${VERSION}"

###############################################################################
; MUI settings
!include "MUI.nsh"

!define MUI_ABORTWARNING
!define MUI_UNABORTWARNING
!define MUI_HEADERIMAGE

; Theme
!define MUI_ICON "%s\\tools\\packaging\\LLVM.ico"
!define MUI_UNICON "%s\\Contrib\\Graphics\\Icons\\orange-uninstall.ico"

!insertmacro MUI_PAGE_WELCOME

!define MUI_LICENSEPAGE_TEXT_BOTTOM "The source code for Cling is freely redistributable under the terms of the GNU Lesser General Public License (LGPL) as published by the Free Software Foundation."
!define MUI_LICENSEPAGE_BUTTON "Next >"
!insertmacro MUI_PAGE_LICENSE "%s"

!insertmacro MUI_PAGE_DIRECTORY

!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_RUN "$INSTDIR\\bin\\${MAIN_APP_EXE}"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM

!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

###############################################################################

Function .onInit
  Call DetectWinVer
  Call CheckPrevVersion
FunctionEnd

; file section
Section "MainFiles"
'''%(prefix,
     VERSION,
     os.path.basename(prefix) + '-setup.exe',
     VIProductVersion.replace('v', ''),
     CLING_SRC_DIR,
     NSIS,
     os.path.join(CLING_SRC_DIR, 'LICENSE.TXT'))

    f.write(template.lstrip())
    f.close()

    # Insert the files to be installed
    f = open(os.path.join(workdir, 'cling.nsi'), 'a+')
    for root, dirs, files in os.walk(prefix):
        f.write(' CreateDirectory "$INSTDIR\\%s"\n'%(root.replace(prefix, '')))
        f.write(' SetOutPath "$INSTDIR\\%s"\n'%(root.replace(prefix, '')))

        for file in files:
            path=os.path.join(root, file)
            f.write(' File "%s"\n'%(path))

    template = '''
SectionEnd

Section make_uninstaller
 ; Write the uninstall keys for Windows
 SetOutPath "$INSTDIR"
 WriteRegStr HKLM "Software\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "DisplayName" "Cling"
 WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "UninstallString" "$INSTDIR\\uninstall.exe"
 WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "NoModify" 1
 WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "NoRepair" 1
 WriteUninstaller "uninstall.exe"
SectionEnd

; start menu
# TODO: This is currently hardcoded.
Section "Shortcuts"

 CreateDirectory "$SMPROGRAMS\\Cling"
 CreateShortCut "$SMPROGRAMS\\Cling\\Uninstall.lnk" "$INSTDIR\\uninstall.exe" "" "$INSTDIR\\uninstall.exe" 0
 CreateShortCut "$SMPROGRAMS\Cling\\Cling.lnk" "$INSTDIR\\bin\\cling.exe" "" "${MUI_ICON}" 0
 CreateDirectory "$SMPROGRAMS\\Cling\\Documentation"
 CreateShortCut "$SMPROGRAMS\\Cling\\Documentation\\Cling (PS).lnk" "$INSTDIR\\docs\\llvm\\ps\\cling.ps" "" "" 0
 CreateShortCut "$SMPROGRAMS\\Cling\\Documentation\\Cling (HTML).lnk" "$INSTDIR\\docs\\llvm\\html\\cling\\cling.html" "" "" 0

SectionEnd

Section "Uninstall"

 DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\\Uninstall\Cling"
 DeleteRegKey HKLM "Software\Cling"

 ; Remove shortcuts
 Delete "$SMPROGRAMS\Cling\*.*"
 Delete "$SMPROGRAMS\Cling\Documentation\*.*"
 Delete "$SMPROGRAMS\Cling\Documentation"
 RMDir "$SMPROGRAMS\Cling"

'''
    f.write(template)

    # insert dir list (depth-first order) for uninstall files
    def walktree (top = prefix):
        names = os.listdir(top)
        for name in names:
            try:
                st = os.lstat(os.path.join(top, name))
            except os.error:
                continue
            if stat.S_ISDIR(st.st_mode):
                for (newtop, children) in walktree (os.path.join(top, name)):
                    yield newtop, children
        yield top, names

    def iterate():
        for (basepath, children) in walktree():
            f.write(' Delete "%s\\*.*"\n'%(basepath.replace(prefix, '$INSTDIR')))
            f.write(' RmDir "%s"\n'%(basepath.replace(prefix, '$INSTDIR')))

    iterate()

    # last bit of the uninstaller
    template = '''
SectionEnd

; Function to detect Windows version and abort if Cling is unsupported in the current platform
Function DetectWinVer
  Push $0
  Push $1
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" CurrentVersion
  IfErrors is_error is_winnt
is_winnt:
  StrCpy $1 $0 1
  StrCmp $1 4 is_error ; Aborting installation for Windows versions older than Windows 2000
  StrCmp $0 "5.0" is_error ; Removing Windows 2000 as supported Windows version
  StrCmp $0 "5.1" is_winnt_XP
  StrCmp $0 "5.2" is_winnt_2003
  StrCmp $0 "6.0" is_winnt_vista
  StrCmp $0 "6.1" is_winnt_7
  StrCmp $0 "6.2" is_winnt_8
  StrCmp $1 6 is_winnt_8 ; Checking for future versions of Windows 8
  Goto is_error

is_winnt_XP:
is_winnt_2003:
is_winnt_vista:
is_winnt_7:
is_winnt_8:
  Goto done
is_error:
  StrCpy $1 $0
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" ProductName
  IfErrors 0 +4
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion" Version
  IfErrors 0 +2
  StrCpy $0 "Unknown"
  MessageBox MB_ICONSTOP|MB_OK "This version of Cling cannot be installed on this system. Cling is supported only on Windows NT systems. Current system: $0 (version: $1)"
  Abort
done:
  Pop $1
  Pop $0
FunctionEnd

; Function to check any previously installed version of Cling in the system
Function CheckPrevVersion
  Push $0
  Push $1
  Push $2
  IfFileExists "$INSTDIR\\bin\cling.exe" 0 otherver
  MessageBox MB_OK|MB_ICONSTOP "Another Cling installation (with the same version) has been detected. Please uninstall it first."
  Abort
otherver:
  StrCpy $0 0
  StrCpy $2 ""
loop:
  EnumRegKey $1 ${PRODUCT_ROOT_KEY} "${PRODUCT_KEY}" $0
  StrCmp $1 "" loopend
  IntOp $0 $0 + 1
  StrCmp $2 "" 0 +2
  StrCpy $2 "$1"
  StrCpy $2 "$2, $1"
  Goto loop
loopend:
  ReadRegStr $1 ${PRODUCT_ROOT_KEY} "${PRODUCT_KEY}" "Version"
  IfErrors finalcheck
  StrCmp $2 "" 0 +2
  StrCpy $2 "$1"
  StrCpy $2 "$2, $1"
finalcheck:
  StrCmp $2 "" done
  MessageBox MB_YESNO|MB_ICONEXCLAMATION "Another Cling installation (version $2) has been detected. It is recommended to uninstall it if you intend to use the same installation directory. Do you want to proceed with the installation anyway?" IDYES done IDNO 0
  Abort
done:
  ClearErrors
  Pop $2
  Pop $1
  Pop $0
FunctionEnd
'''
    f.write(template)
    f.close()

def build_nsis():
    box_draw("Build NSIS executable from cling.nsi")
    NSIS = os.path.join(TMP_PREFIX, 'bin', 'nsis')
    exec_subprocess_call('%s -V3 %s'%(os.path.join(NSIS, 'makensis.exe'), os.path.join(workdir, 'cling.nsi')), workdir)

###############################################################################
#                          Mac OS X specific functions                        #
###############################################################################

def check_mac(pkg):
    if pkg == "python":
        if platform.python_version()[0] == '3':
            print(pkg.ljust(20) + '[UNSUPPORTED VERSION (Python 3)]'.ljust(30))
        elif float(platform.python_version()[:3]) < 2.7:
            print(pkg.ljust(20) + '[OUTDATED VERSION (<2.7)]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))
    elif pkg == "SSL":
        import socket
        if hasattr(socket, 'ssl'):
            print(pkg.ljust(20) + '[SUPPORTED]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[NOT SUPPORTED]'.ljust(30))
    elif exec_subprocess_check_output("type -p %s"%(pkg), '/').strip() == '':
        print(pkg.ljust(20) + '[NOT INSTALLED]'.ljust(30))
    else:
        if pkg == "g++":
            if float(exec_subprocess_check_output('g++ -dumpversion', '/')[:3].strip()) <= 4.7:
                print(pkg.ljust(20) + '[UNSUPPORTED VERSION (<4.7)]'.ljust(30))
            else:
                print(pkg.ljust(20) + '[OK]'.ljust(30))
        elif pkg == "gcc":
            if float(exec_subprocess_check_output('gcc -dumpversion', '/')[:3].strip()) <= 4.7:
                print(pkg.ljust(20) + '[UNSUPPORTED VERSION (<4.7)]'.ljust(30))
            else:
                print(pkg.ljust(20) + '[OK]'.ljust(30))
        else:
            print(pkg.ljust(20) + '[OK]'.ljust(30))


def make_dmg():
    box_draw("Building Apple Disk Image")
    APP_NAME = 'Cling'
    DMG_BACKGROUND_IMG = 'Background.png'
    APP_EXE = '%s.app/Contents/MacOS/%s'%(APP_NAME, APP_NAME)
    VOL_NAME = "%s-%s"%(APP_NAME.lower(), VERSION)
    DMG_TMP = "%s-temp.dmg"%(VOL_NAME)
    DMG_FINAL = "%s.dmg"%(VOL_NAME)
    STAGING_DIR = os.path.join(workdir, 'Install')

    if os.path.isdir(STAGING_DIR):
        print("Remove directory: " + STAGING_DIR)
        shutil.rmtree(STAGING_DIR)

    if os.path.isdir(os.path.join(workdir, '%s.app'%(APP_NAME))):
        print("Remove directory: " + os.path.join(workdir, '%s.app'%(APP_NAME)))
        shutil.rmtree(os.path.join(workdir, '%s.app'%(APP_NAME)))

    if os.path.isdir(os.path.join(workdir, DMG_TMP)):
        print("Remove directory: " + os.path.join(workdir, DMG_TMP))
        shutil.rmtree(os.path.join(workdir, DMG_TMP))

    if os.path.isdir(os.path.join(workdir, DMG_FINAL)):
        print("Remove directory: " + os.path.join(workdir, DMG_FINAL))
        shutil.rmtree(os.path.join(workdir, DMG_FINAL))

    print('Create directory: ' + os.path.join(workdir, '%s.app'%(APP_NAME)))
    os.makedirs(os.path.join(workdir, '%s.app'%(APP_NAME)))

    print('Populate directory: ' + os.path.join(workdir, '%s.app/Contents/Resources'%(APP_NAME)))
    shutil.copytree(prefix, os.path.join(workdir, '%s.app/Contents/Resources'%(APP_NAME)))

    print('Copy APP Bundle to staging area: ' + STAGING_DIR)
    shutil.copytree(os.path.join(workdir,'%s.app'%(APP_NAME)), STAGING_DIR)

    print('Stripping file: ' + APP_EXE.lower())
    exec_subprocess_call('strip -u -r %s'%(APP_EXE.lower()), workdir)

    DU = exec_subprocess_check_output("du -sh %s"%(STAGING_DIR), workdir)
    SIZE = str(float(DU[:DU.find('M')].strip()) + 1.0)
    print('Estimated size of application bundle: ' + SIZE + 'MB')

    print('Building temporary Apple Disk Image')
    exec_subprocess_call('hdiutil create -srcfolder %s -volname %s -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDRW -size %sM %s'%(STAGING_DIR, VOL_NAME, SIZE, DMG_TMP), workdir)

    print('Created Apple Disk Image: ' + DMG_TMP)
    DEVICE = exec_subprocess_check_output("hdiutil attach -readwrite -noverify -noautoopen %s | egrep '^/dev/' | sed 1q | awk '{print $1}'"%(DMG_TMP), workdir)

    print('Wating for device to unmount...')
    time.sleep(5)

    #print 'Create directory: ' + '/Volumes/%s/.background'%(VOL_NAME)
    #os.makedirs('/Volumes/%s/.background'%(VOL_NAME))
    #shutil.copy(os.path.join(workdir,DMG_BACKGROUND_IMG), '/Volumes/%s/.background/'%(VOL_NAME))

    ascript = '''
tell application "Finder"
  tell disk "%s"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {400, 100, 920, 440}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 72
        set background picture of viewOptions to file ".background:%s"
        set position of item "%s.app" of container window to {160, 205}
        set position of item "Applications" of container window to {360, 205}
        close
        open
        update without registering application
        delay 2
  end tell
end tell
'''%(VOL_NAME, DMG_BACKGROUND_IMG, APP_NAME)
    ascript = ascript.strip()

    print('Executing AppleScript...')
    exec_subprocess_call("echo %s | osascript"%(ascript), workdir)

    print('Performing sync...')
    exec_subprocess_call("sync", workdir)

    print('Detach device: ' + DEVICE)
    exec_subprocess_call('hdiutil detach %s'%(DEVICE), CLING_SRC_DIR)

    print("Creating compressed Apple Disk Image...")
    exec_subprocess_call('hdiutil convert %s -format UDZO -imagekey zlib-level=9 -o %s'%(DMG_TMP, DMG_FINAL), workdir)

    print('Done')

###############################################################################
#                           argparse configuration                            #
###############################################################################

parser = argparse.ArgumentParser(description='Cling Packaging Tool')
parser.add_argument('-c', '--check-requirements', help='Check if packages required by the script are installed', action='store_true')
parser.add_argument('--current-dev', help='Package the latest development snapshot in one of these formats: tar | deb | nsis | rpm | dmg | pkg')
parser.add_argument('--last-stable', help='Package the last stable snapshot in one of these formats: tar | deb | nsis | rpm | dmg | pkg')
parser.add_argument('--tarball-tag', help='Package the snapshot of a given tag in a tarball (.tar.bz2)')
parser.add_argument('--deb-tag', help='Package the snapshot of a given tag in a Debian package (.deb)')
parser.add_argument('--rpm-tag', help='Package the snapshot of a given tag in an RPM package (.rpm)')
parser.add_argument('--nsis-tag', help='Package the snapshot of a given tag in an NSIS installer (.exe)')
parser.add_argument('--dmg-tag', help='Package the snapshot of a given tag in a DMG package (.dmg)')

# Variable overrides
parser.add_argument('--with-llvm-url', action='store', help='Specify an alternate URL of LLVM repo', default='http://root.cern.ch/git/llvm.git')
parser.add_argument('--with-clang-url', action='store', help='Specify an alternate URL of Clang repo', default='http://root.cern.ch/git/clang.git')
parser.add_argument('--with-cling-url', action='store', help='Specify an alternate URL of Cling repo', default='http://root.cern.ch/git/cling.git')

parser.add_argument('--no-test', help='Do not run test suite of Cling', action='store_true')
parser.add_argument('--create-dev-env', help='Set up a release/debug environment')

if platform.system() != 'Windows':
    parser.add_argument('--with-workdir', action='store', help='Specify an alternate working directory for CPT', default=os.path.expanduser(os.path.join('~', 'ec', 'build')))
else:
    parser.add_argument('--with-workdir', action='store', help='Specify an alternate working directory for CPT', default='C:\\ec\\build\\')

parser.add_argument('--make-proper', help='Internal option to support calls from build system')

args = vars(parser.parse_args())

###############################################################################
#                           Platform initialization                           #
###############################################################################

OS=platform.system()
FAMILY=os.name.upper()

if OS == 'Windows':
    DIST = 'N/A'
    RELEASE = OS + ' ' + platform.release()
    REV = platform.version()

    EXEEXT = '.exe'
    SHLIBEXT = '.dll'

    TMP_PREFIX='C:\\Windows\\Temp\\cling-obj\\'

elif OS == 'Linux':
    DIST = platform.linux_distribution()[0]
    RELEASE = platform.linux_distribution()[2]
    REV = platform.linux_distribution()[1]

    EXEEXT = ''
    SHLIBEXT = '.so'

    TMP_PREFIX=os.path.join(os.sep, 'tmp', 'cling-obj' + os.sep)

elif OS == 'Darwin':
    DIST = 'MacOSX'
    RELEASE = platform.release()
    REV = platform.mac_ver()[0]

    EXEEXT = ''
    SHLIBEXT = '.dylib'

    TMP_PREFIX=os.path.join(os.sep, 'tmp', 'cling-obj' + os.sep)

else:
    # Extensions will be detected anyway by set_ext()
    EXEEXT = ''
    SHLIBEXT = ''

    #TODO: Need to test this in other platforms
    TMP_PREFIX=os.path.join(os.sep, 'tmp', 'cling-obj' + os.sep)

###############################################################################
#                               Global variables                              #
###############################################################################

workdir = os.path.abspath(os.path.expanduser(args['with_workdir']))

# This is needed in Windows
if not os.path.isdir(workdir):
    os.makedirs(workdir)

if os.path.isdir(TMP_PREFIX):
    shutil.rmtree(TMP_PREFIX)

os.makedirs(TMP_PREFIX)

srcdir = os.path.join(workdir, 'cling-src')
CLING_SRC_DIR = os.path.join(srcdir, 'tools', 'cling')
LLVM_OBJ_ROOT = os.path.join(workdir, 'builddir')
prefix = ''
LLVM_GIT_URL = args['with_llvm_url']
CLANG_GIT_URL = args['with_clang_url']
CLING_GIT_URL = args['with_cling_url']
LLVMRevision = urlopen("https://raw.githubusercontent.com/vgvassilev/cling/master/LastKnownGoodLLVMSVNRevision.txt").readline().strip().decode('utf-8')
VERSION = ''
REVISION = ''

print('Cling Packaging Tool (CPT)')
print('Arguments vector: ' + str(sys.argv))
box_draw_header()
print('Thread Model: ' + FAMILY)
print('Operating System: ' + OS)
print('Distribution: ' + DIST)
print('Release: ' + RELEASE)
print('Revision: ' + REV)
print('Architecture: ' + platform.machine())

if len(sys.argv) == 1:
    print("Error: no options passed")
    parser.print_help()

if args['check_requirements'] == True:
    box_draw('Check availability of required softwares')
    if DIST == 'Ubuntu':
        check_ubuntu('git')
        check_ubuntu('gcc')
        check_ubuntu('g++')
        check_ubuntu('debhelper')
        check_ubuntu('devscripts')
        check_ubuntu('gnupg')
        check_ubuntu('python')
        check_ubuntu('SSL')
        yes = set(['yes','y', 'ye', ''])
        no = set(['no','n'])

        choice = input('''
CPT will now attempt to update/install the requisite packages automatically.
Do you want to continue? [yes/no]: ''').lower()
        while True:
            if choice in yes:
                # Need to communicate values to the shell. Do not use exec_subprocess_call()
                subprocess.Popen(['sudo apt-get update'],
                                 shell=True,
                                 stdin=subprocess.PIPE,
                                 stdout=None,
                                 stderr=subprocess.STDOUT).communicate('yes')
                subprocess.Popen(['sudo apt-get install git g++ debhelper devscripts gnupg python'],
                                  shell=True,
                                  stdin=subprocess.PIPE,
                                  stdout=None,
                                  stderr=subprocess.STDOUT).communicate('yes')
                break
            elif choice in no:
                print('''
Install/update the required packages by:
  sudo apt-get update
  sudo apt-get install git g++ debhelper devscripts gnupg python
''')
                break
            else:
                choice = input("Please respond with 'yes' or 'no': ")
                continue

    elif OS == 'Windows':
        check_win('git')
        check_win('python')
        check_win('SSL')
        # Check Windows registry for keys that prove an MS Visual Studio 11.0 installation
        check_win('msvc')
        print('''
Refer to the documentation of CPT for information on setting up your Windows environment.
[tools/packaging/README.md]
''')
    elif DIST == 'Fedora' or DIST == 'Scientific Linux CERN SLC':
        check_redhat('git')
        check_redhat('gcc')
        check_redhat('gcc-c++')
        check_redhat('rpm-build')
        check_redhat('python')
        check_redhat('SSL')
        yes = set(['yes','y', 'ye', ''])
        no = set(['no','n'])

        choice = input('''
CPT will now attempt to update/install the requisite packages automatically.
Do you want to continue? [yes/no]: ''').lower()
        while True:
            if choice in yes:
                # Need to communicate values to the shell. Do not use exec_subprocess_call()
                subprocess.Popen(['sudo yum install git gcc gcc-c++ rpm-build python'],
                                  shell=True,
                                  stdin=subprocess.PIPE,
                                  stdout=None,
                                  stderr=subprocess.STDOUT).communicate('yes')
                break
            elif choice in no:
                print('''
Install/update the required packages by:
  sudo yum install git gcc gcc-c++ rpm-build python
''')
                break
            else:
                choice = input("Please respond with 'yes' or 'no': ")
                continue

    if DIST == 'MacOSX':
        check_mac('git')
        check_mac('gcc')
        check_mac('g++')
        check_mac('python')
        check_mac('SSL')
        yes = set(['yes','y', 'ye', ''])
        no = set(['no','n'])

        choice = input('''
CPT will now attempt to update/install the requisite packages automatically. Make sure you have MacPorts installed.
Do you want to continue? [yes/no]: ''').lower()
        while True:
            if choice in yes:
                # Need to communicate values to the shell. Do not use exec_subprocess_call()
                subprocess.Popen(['sudo port -v selfupdate'],
                                 shell=True,
                                 stdin=subprocess.PIPE,
                                 stdout=None,
                                 stderr=subprocess.STDOUT).communicate('yes')
                subprocess.Popen(['sudo port install git g++ python'],
                                  shell=True,
                                  stdin=subprocess.PIPE,
                                  stdout=None,
                                  stderr=subprocess.STDOUT).communicate('yes')
                break
            elif choice in no:
                print('''
Install/update the required packages by:
  sudo port -v selfupdate
  sudo port install git g++ python
''')
                break
            else:
                choice = input("Please respond with 'yes' or 'no': ")
                continue

if args['current_dev']:
    fetch_llvm()
    fetch_clang()
    fetch_cling('master')
    set_version()
    if args['current_dev'] == 'tar':
        if OS == 'Windows':
            get_win_dep()
            compile(os.path.join(workdir, 'cling-win-' + platform.machine().lower() + '-' + VERSION))
        else:
            if DIST == 'Scientific Linux CERN SLC':
                compile(os.path.join(workdir, 'cling-SLC-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
            else:
                compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        tarball()
        cleanup()

    elif args['current_dev'] == 'deb' or (args['current_dev'] == 'pkg' and DIST == 'Ubuntu'):
        compile(os.path.join(workdir, 'cling-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        tarball_deb()
        debianize()
        cleanup()

    elif args['current_dev'] == 'rpm' or (args['current_dev'] == 'pkg' and platform.dist()[0] == 'redhat'):
        compile(os.path.join(workdir, 'cling-' + VERSION.replace('-' + REVISION[:7], '')))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        tarball()
        rpm_build()
        cleanup()

    elif args['current_dev'] == 'nsis' or (args['current_dev'] == 'pkg' and OS == 'Windows'):
        get_win_dep()
        compile(os.path.join(workdir, 'cling-' + RELEASE + '-' + platform.machine().lower() + '-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        make_nsi()
        build_nsis()
        cleanup()

    elif args['current_dev'] == 'dmg' or (args['current_dev'] == 'pkg' and OS == 'Darwin'):
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        make_dmg()
        cleanup()

if args['last_stable']:
    fetch_llvm()
    fetch_clang()
    fetch_cling('last-stable')

    if args['last_stable'] == 'tar':
        set_version()
        if OS == 'Windows':
            get_win_dep()
            compile(os.path.join(workdir, 'cling-win-' + platform.machine().lower() + '-' + VERSION))
        else:
            if DIST == 'Scientific Linux CERN SLC':
                compile(os.path.join(workdir, 'cling-SLC-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
            else:
                compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        tarball()
        cleanup()

    elif args['last_stable'] == 'deb' or (args['last_stable'] == 'pkg' and DIST == 'Ubuntu'):
        set_version()
        compile(os.path.join(workdir, 'cling-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        tarball_deb()
        debianize()
        cleanup()

    elif args['last_stable'] == 'rpm' or (args['last_stable'] == 'pkg' and platform.dist()[0] == 'redhat'):
        set_version()
        compile(os.path.join(workdir, 'cling-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        tarball()
        rpm_build()
        cleanup()

    elif args['last_stable'] == 'nsis' or (args['last_stable'] == 'pkg' and OS == 'Windows'):
        set_version()
        get_win_dep()
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        make_nsi
        build_nsis()
        cleanup()

    elif args['last_stable'] == 'dmg' or (args['last_stable'] == 'pkg' and OS == 'Darwin'):
        set_version()
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
        install_prefix()
        if args['no_test'] != True:
            test_cling()
        make_dmg()
        cleanup()

if args['tarball_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['tarball_tag'])
    set_version()

    if OS == 'Windows':
        get_win_dep()
        compile(os.path.join(workdir, 'cling-win-' + platform.machine().lower() + '-' + VERSION))
    else:
        if DIST == 'Scientific Linux CERN SLC':
            compile(os.path.join(workdir, 'cling-SLC-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
        else:
            compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))

    install_prefix()
    if args['no_test'] != True:
        test_cling()
    tarball()
    cleanup()

if args['deb_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['deb_tag'])
    set_version()
    compile(os.path.join(workdir, 'cling-' + VERSION))
    install_prefix()
    if args['no_test'] != True:
        test_cling()
    tarball_deb
    debianize
    cleanup()

if args['rpm_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['rpm_tag'])
    set_version()
    compile(os.path.join(workdir, 'cling-' + VERSION))
    install_prefix()
    if args['no_test'] != True:
        test_cling()
    tarball()
    rpm_build()
    cleanup()

if args['nsis_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['nsis_tag'])
    set_version()
    get_win_dep()
    compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
    install_prefix()
    if args['no_test'] != True:
        test_cling()
    make_nsi
    build_nsis()
    cleanup()

if args['dmg_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['dmg_tag'])
    set_version()
    compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
    install_prefix()
    if args['no_test'] != True:
        test_cling()
    make_dmg()
    cleanup()

if args['create_dev_env']:
    fetch_llvm()
    fetch_clang()
    fetch_cling('master')
    set_version()
    if OS == 'Windows':
        get_win_dep()
        compile(os.path.join(workdir, 'cling-win-' + platform.machine().lower() + '-' + VERSION))
    else:
        if DIST == 'Scientific Linux CERN SLC':
            compile(os.path.join(workdir, 'cling-SLC-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
        else:
            compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine().lower() + '-' + VERSION))
    install_prefix()
    if args['no_test'] != True:
        test_cling()

if args['make_proper']:
    # This is an internal option in CPT, meant to be integrated into Cling's build system.
    with open(os.path.join(LLVM_OBJ_ROOT, 'config.log'), 'r') as log:
        for line in log:
            if re.match('^LLVM_PREFIX=', line):
                prefix=re.sub('^LLVM_PREFIX=', '', line).replace("'", '').strip()

    set_version()
    install_prefix()
    cleanup()
