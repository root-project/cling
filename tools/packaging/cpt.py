#! /usr/bin/env python
# coding:utf-8

###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/cpt.py: Main script which calls other helper scripts to
# compile and package Cling for multiple platforms.
#
# TODO: Add documentation here, or provide link to documentation
#
# Author: Anirudha Bose <ani07nov@gmail.com>
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#
###############################################################################

import argparse
import urllib2
import os
import sys
import platform
import subprocess
import shutil
import glob
import re
import tarfile
from email.utils import formatdate
from datetime import tzinfo
import time
import multiprocessing
import fileinput

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

    TMP_PREFIX='C:\\Windows\\Temp\\cling-obj'
    workdir = 'C:\\ec\\build'

elif OS == 'Linux':
    DIST = platform.linux_distribution()[0]
    RELEASE = platform.linux_distribution()[2]
    REV = platform.linux_distribution()[1]

    EXEEXT = ''
    SHLIBEXT = '.so'

    TMP_PREFIX=os.path.join(os.sep, 'var', 'tmp', 'cling-obj' + os.sep)
    workdir = os.path.expanduser(os.path.join('~', 'ec', 'build'))

elif OS == 'Darwin':
    DIST = 'N/A'
    RELEASE = platform.release()
    REV = platform.mac_ver()[0]

    EXEEXT = ''
    SHLIBEXT = '.dylib'

    TMP_PREFIX=os.path.join(os.sep, 'var', 'tmp', 'cling-obj' + os.sep)
    workdir = os.path.expanduser(os.path.join('~', 'ec', 'build'))

else:
    # Extensions will be detected anyway by set_ext()
    EXEEXT = ''
    SHLIBEXT = ''

    #TODO: Need to test this in other platforms
    TMP_PREFIX=os.path.join(os.sep, 'var', 'tmp', 'cling-obj' + os.sep)
    workdir = os.path.expanduser(os.path.join('~', 'ec', 'build'))


###############################################################################
#                               Global variables                              #
###############################################################################

srcdir = os.path.join(workdir, 'cling-src')
CLING_SRC_DIR = os.path.join(srcdir, 'tools', 'cling')
LLVM_OBJ_ROOT = os.path.join(workdir, 'builddir')
prefix = ''
LLVM_GIT_URL = 'http://root.cern.ch/git/llvm.git'
CLANG_GIT_URL = 'http://root.cern.ch/git/clang.git'
CLING_GIT_URL = 'http://root.cern.ch/git/cling.git'
LLVMRevision = urllib2.urlopen("https://raw.githubusercontent.com/ani07nov/cling/master/LastKnownGoodLLVMSVNRevision.txt").readline().strip()
VERSION=''


###############################################################################
#              Platform independent functions (formerly indep.py)             #
###############################################################################

def exec_subprocess_call(cmd, cwd):
    if OS == 'Windows':
        if '"' not in cmd:
            subprocess.Popen(cmd.split(),
                             cwd=cwd,
                             shell=True,
                             stdin=subprocess.PIPE,
                             stdout=None,
                             stderr=subprocess.STDOUT).communicate()
        else:
            subprocess.Popen(cmd.split('"')[0].split() + [cmd.split('"')[1]] + cmd.split('"')[2].split(),
                             cwd=cwd,
                             shell=True,
                             stdin=subprocess.PIPE,
                             stdout=None,
                             stderr=subprocess.STDOUT).communicate()
    else:
        subprocess.Popen([cmd],
                         cwd=cwd,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT).communicate()

def exec_subprocess_check_output(cmd, cwd):
    if OS == 'Windows':
        return subprocess.Popen(cmd.split(),
                     cwd=cwd,
                     shell=True,
                     stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT).communicate()[0]
    else:
        return subprocess.Popen([cmd],
                     cwd=cwd,
                     shell=True,
                     stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT).communicate()[0]


def box_draw_header():
    msg='cling (' + platform.machine() + ')' + formatdate(time.time(),tzinfo())
    spaces_no = 80 - len(msg) - 4
    spacer = ' ' * spaces_no
    msg='cling (' + platform.machine() + ')' + spacer + formatdate(time.time(),tzinfo())

    if OS != 'Windows':
        print '''
╔══════════════════════════════════════════════════════════════════════════════╗
║ %s ║
╚══════════════════════════════════════════════════════════════════════════════╝'''%(msg)
    else:
        print '''
+=============================================================================+
| %s|
+=============================================================================+'''%(msg)


def box_draw(msg):
    spaces_no = 80 - len(msg) - 4
    spacer = ' ' * spaces_no

    if OS != 'Windows':
        print '''
┌──────────────────────────────────────────────────────────────────────────────┐
│ %s%s │
└──────────────────────────────────────────────────────────────────────────────┘'''%(msg, spacer)
    else:
        print '''
+-----------------------------------------------------------------------------+
| %s%s|
+-----------------------------------------------------------------------------+'''%(msg, spacer)


def fetch_llvm():
    box_draw("Fetch source files")
    print "Last known good LLVM revision is: %s"%(LLVMRevision)
    def get_fresh_llvm():
        exec_subprocess_call('git clone %s %s'%(LLVM_GIT_URL, srcdir), workdir)

        exec_subprocess_call('git checkout ROOT-patches-r%s'%(LLVMRevision), srcdir)

    def update_old_llvm():
        exec_subprocess_call('git stash', srcdir)

        exec_subprocess_call('git clean -f -x -d', srcdir)

        exec_subprocess_call('git fetch --tags', srcdir)

        exec_subprocess_call('git checkout ROOT-patches-r%s'%(LLVMRevision), srcdir)

        exec_subprocess_call('git pull origin refs/tags/ROOT-patches-r%s'%(LLVMRevision), srcdir)

    if os.path.isdir(srcdir):
        update_old_llvm()
    else:
        get_fresh_llvm()


def fetch_clang():
    def get_fresh_clang():
        exec_subprocess_call('git clone %s'%(CLANG_GIT_URL), os.path.join(srcdir, 'tools'))

        exec_subprocess_call('git checkout ROOT-patches-r%s'%(LLVMRevision), os.path.join(srcdir, 'tools', 'clang'))

    def update_old_clang():
        exec_subprocess_call('git stash', os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git clean -f -x -d', os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git fetch --tags', os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git checkout ROOT-patches-r%s'%(LLVMRevision), os.path.join(srcdir, 'tools', 'clang'))

        exec_subprocess_call('git pull origin refs/tags/ROOT-patches-r%s'%(LLVMRevision), os.path.join(srcdir, 'tools', 'clang'))

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
    box_draw("Set Cling version")
    VERSION=open(os.path.join(CLING_SRC_DIR, 'VERSION'), 'r').readline().strip()

    # If development release, then add revision to the version
    REVISION = exec_subprocess_check_output('git log -n 1 --pretty=format:%H', CLING_SRC_DIR).strip()

    if '~dev' in VERSION:
        VERSION = VERSION + '-' + REVISION[:7]

    print 'Version: ' + VERSION
    print 'Revision: ' + REVISION


def set_ext():
    global EXEEXT
    global SHLIBEXT
    box_draw("Set binary/library extensions")
    if not os.path.isfile(os.path.join(LLVM_OBJ_ROOT, 'test', 'lit.site.cfg')):
        exec_subprocess_call('make lit.site.cfg', os.path.join(LLVM_OBJ_ROOT, 'test'))

    with open(os.path.join(LLVM_OBJ_ROOT, 'test', 'lit.site.cfg'), 'r') as lit_site_cfg:
        for line in lit_site_cfg:
            if re.match('^config.llvm_shlib_ext = ', line):
                SHLIBEXT = re.sub('^config.llvm_shlib_ext = ', '', line).replace('"', '').strip()
            elif re.match('^config.llvm_exe_ext = ', line):
                EXEEXT = re.sub('^config.llvm_exe_ext = ', '', line).replace('"', '').strip()

    print 'EXEEXT: ' + EXEEXT
    print 'SHLIBEXT: ' + SHLIBEXT


def compile(arg):
    global prefix
    prefix=arg
    python=sys.executable
    cores=multiprocessing.cpu_count()

    # Cleanup previous installation directory if any
    if os.path.isdir(prefix):
        print "Remove directory: " + prefix
        shutil.rmtree(prefix)

    # Cleanup previous build directory if exists
    if os.path.isdir(os.path.join(workdir, 'builddir')):
        print "Remove directory: " + os.path.join(workdir, 'builddir')
        shutil.rmtree(os.path.join(workdir, 'builddir'))

    os.makedirs(os.path.join(workdir, 'builddir'))

    if platform.system() == 'Windows':
        box_draw("Configure Cling with CMake and generate Visual Studio 11 project files")
        exec_subprocess_call('cmake -G "Visual Studio 11" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%s ..\%s'%(TMP_PREFIX, os.path.basename(srcdir)), LLVM_OBJ_ROOT)

        box_draw("Building Cling (using %s cores)"%(cores))
        exec_subprocess_call('cmake --build . --target clang --config Release', LLVM_OBJ_ROOT)

        exec_subprocess_call('cmake --build . --target cling --config Release', LLVM_OBJ_ROOT)

        box_draw("Install compiled binaries to prefix (using %s cores)"%(cores))
        exec_subprocess_call('cmake --build . --target INSTALL --config Release', LLVM_OBJ_ROOT)

    else:
        box_draw("Configure Cling with GNU Make")
        exec_subprocess_call('%s/configure --disable-compiler-version-checks --with-python=%s --enable-targets=host --prefix=%s --enable-optimized=yes --enable-cxx11'%(srcdir, python, TMP_PREFIX), LLVM_OBJ_ROOT)

        box_draw("Building Cling (using %s cores)"%(cores))
        exec_subprocess_call('make -j%s'%(cores), LLVM_OBJ_ROOT)

        box_draw("Install compiled binaries to prefix (using %s cores)"%(cores))
        exec_subprocess_call('make install -j%s prefix=%s'%(cores, TMP_PREFIX), LLVM_OBJ_ROOT)


def install_prefix():
    set_ext()
    box_draw("Filtering Cling's libraries and binaries")

    for line in fileinput.input(os.path.join(CLING_SRC_DIR, 'tools', 'packaging', 'dist-files.mk'), inplace=True):
        if '@EXEEXT@' in line:
            print line.replace('@EXEEXT@', EXEEXT),
        elif '@SHLIBEXT@' in line:
            print line.replace('@SHLIBEXT@', SHLIBEXT),
        else:
            print line,

    dist_files = open(os.path.join(CLING_SRC_DIR, 'tools', 'packaging', 'dist-files.mk'), 'r').read()
    for root, dirs, files in os.walk(TMP_PREFIX):
        for file in files:
            f=os.path.join(root, file).replace(TMP_PREFIX, '')
            if f+' ' in dist_files:
                print "Filter: " + f
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
    tar.add(prefix, arcname=os.path.basename(prefix))
    tar.close()


def cleanup():
    print "\n"
    box_draw("Clean up")
    if os.path.isdir(os.path.join(workdir, 'builddir')):
        print "Remove directory: " + os.path.join(workdir, 'builddir')
        shutil.rmtree(os.path.join(workdir, 'builddir'))

    if os.path.isdir(prefix):
        print "Remove directory: " + prefix
        shutil.rmtree(prefix)

    if os.path.isdir(TMP_PREFIX):
        print "Remove directory: " + TMP_PREFIX
        shutil.rmtree(TMP_PREFIX)

    if os.path.isfile(os.path.join(workdir,'cling.nsi')):
        print "Remove file: " + os.path.join(workdir,'cling.nsi')
        os.remove(os.path.join(workdir,'cling.nsi'))

    if args['current_dev'] == 'deb' or args['last_stable'] == 'deb' or args['deb_tag']:
        print 'Create output directory: ' + os.path.join(workdir, 'cling-%s-1'%(VERSION))
        os.makedirs(os.path.join(workdir, 'cling-%s-1'%(VERSION)))

        for file in glob.glob(os.path.join(workdir, 'cling_%s*'%(VERSION))):
            print file + '->' + os.path.join(workdir, 'cling-%s-1'%(VERSION), os.path.basename(file))
            shutil.move(file, os.path.join(workdir, 'cling-%s-1'%(VERSION)))

        if not os.listdir(os.path.join(workdir, 'cling-%s-1'%(VERSION))):
            os.rmdir(os.path.join(workdir, 'cling-%s-1'%(VERSION)))

###############################################################################
#            Debian specific functions (ported from debianize.sh)             #
###############################################################################

def tarball_deb():
    box_draw("Compress compiled binaries into a bzip2 tarball")
    tar = tarfile.open(os.path.join(workdir, 'cling_' + VERSION +'.orig.tar.bz2'), 'w:bz2')
    tar.add(prefix, arcname=os.path.basename(prefix))
    tar.close()

def debianize():
    SIGNING_USER = exec_subprocess_check_output('gpg --fingerprint | grep uid | sed s/"uid *"//g', CLING_SRC_DIR).strip()

    box_draw("Set up the debian directory")
    print "Create directory: debian"
    os.makedirs(os.path.join(prefix, 'debian'))

    print "Create directory: " + os.path.join(prefix, 'debian', 'source')
    os.makedirs(os.path.join(prefix, 'debian', 'source'))

    print "Create file: " + os.path.join(prefix, 'debian', 'source', 'format')
    f=open(os.path.join(prefix, 'debian', 'source', 'format'), 'w')
    f.write('3.0 (quilt)')
    f.close()

    print "Create file: " + os.path.join(prefix, 'debian', 'source', 'lintian-overrides')
    f=open(os.path.join(prefix, 'debian', 'source', 'lintian-overrides'), 'w')
    f.write('cling source: source-is-missing')
    f.close()

    # This section is no longer valid. I have kept it as a reference if we plan to
    # distribute libcling.so or any other library with the package.
    if False:
        print 'Create file: ' + os.path.join(prefix, 'debian', 'postinst')
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

    print 'Create file: ' + os.path.join(prefix, 'debian', 'cling.install')
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

    print 'Create file: ' + os.path.join(prefix, 'debian', 'compact')
    # Optimize binary compression
    f = open(os.path.join(prefix, 'debian', 'compact'), 'w')
    f.write("7")
    f.close()

    print 'Create file: ' + os.path.join(prefix, 'debian', 'compat')
    f = open(os.path.join(prefix, 'debian', 'compat'), 'w')
    f.write("9")
    f.close()

    print 'Create file: ' + os.path.join(prefix, 'debian', 'control')
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

    print 'Create file: ' + os.path.join(prefix, 'debian', 'copyright')
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

    print 'Create file: ' + os.path.join(prefix, 'debian', 'rules')
    f = open(os.path.join(prefix, 'debian', 'rules'), 'w')
    template = '''
#!/usr/bin/make -f
# -*- makefile -*-

%:
	dh \$@

override_dh_auto_build:

override_dh_auto_install:
'''
    f.write(template.strip())
    f.close()

    print 'Create file: ' + os.path.join(prefix, 'debian', 'changelog')
    f = open(os.path.join(prefix, 'debian', 'changelog'), 'w')

    template = '''
cling (%s-1) unstable; urgency=low

  * [Debian] Upload to unstable for version: %s
'''%(VERSION)
    f.write(template.ltrip())
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
    f.write(template)
    f.close()

    box_draw("Run debuild to create Debian package")
    exec_subprocess_call('debuild', prefix)


###############################################################################
#                           argparse configuration                            #
###############################################################################

parser = argparse.ArgumentParser(description='Cling Packaging Tool')
parser.add_argument('-c', '--check-requirements', help='Check if packages required by the script are installed', action='store_true')
parser.add_argument('--current-dev', help='Package the latest development snapshot in one of these formats: tar | deb | nsis')
parser.add_argument('--last-stable', help='Package the last stable snapshot in one of these formats: tar | deb | nsis')
parser.add_argument('--tarball-tag', help='Package the snapshot of a given tag in a tarball (.tar.bz2)')
parser.add_argument('--deb-tag', help='Package the snapshot of a given tag in a Debian package (.deb)')
parser.add_argument('--nsis-tag', help='Package the snapshot of a given tag in an NSIS installer (.exe)')

# Variable overrides
parser.add_argument('--with-llvm-url', help='Specify an alternate URL of LLVM repo', default='http://root.cern.ch/git/llvm.git')
parser.add_argument('--with-clang-url', help='Specify an alternate URL of Clang repo', default='http://root.cern.ch/git/clang.git')
parser.add_argument('--with-cling-url', help='Specify an alternate URL of Cling repo', default='http://root.cern.ch/git/cling.git')
parser.add_argument('--with-workdir', help='Specify an alternate working directory for CPT', default=os.path.expanduser(workdir))

args = vars(parser.parse_args())


print 'Cling Packaging Tool (CPT)'
print 'Arguments vector: ' + str(sys.argv)
box_draw_header()
print 'Family: ' + FAMILY
print 'Operating System: ' + OS
print 'Distribution: ' + DIST
print 'Release: ' + RELEASE
print 'Revision: ' + REV + '\n'

if len(sys.argv) == 1:
    print "Error: no options passed"
    parser.print_help()

if args['current_dev']:
    fetch_llvm()
    fetch_clang()
    fetch_cling('master')
    set_version()
    if args['current_dev'] == 'tar':
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
        install_prefix()
        test_cling()
        tarball()
        cleanup()

    elif args['current_dev'] == "deb":
        compile(os.path.join(workdir, 'cling-' + VERSION))
        install_prefix()
        test_cling()
        tarball_deb()
        debianize()
        cleanup()
    elif args['current_dev'] == 'nsis':
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
        install_prefix()
        test_cling()
        #get_nsis
        #make_nsi
        #build_nsis
        cleanup()
 

if args['last_stable']:
    fetch_llvm()
    fetch_clang()
    fetch_cling('last-stable')

    if args['last_stable'] == 'tar':
        set_version()
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
        install_prefix()
        test_cling()
        tarball()
        cleanup()
    if args['last_stable'] == 'deb':
        set_version()
        compile(os.path.join(workdir, 'cling-' + VERSION))
        install_prefix()
        test_cling()
        tarball_deb()
        debianize()
        cleanup()
    if args['last_stable'] == 'nsis':
        compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
        install_prefix()
        test_cling()
        #get_nsis
        #make_nsi
        #build_nsis
        cleanup()

if args['tarball_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['tarball_tag'])
    set_version()
    compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
    install_prefix()
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
    test_cling()
    tarball_deb
    debianize
    cleanup()

if args['nsis_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['nsis_tag'])
    set_version()
    compile(os.path.join(workdir, 'cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION))
    install_prefix()
    test_cling()
    #get_nsis
    #make_nsi
    #build_nsis
    cleanup()
