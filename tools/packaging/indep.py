#! /usr/bin/env python
# coding:utf-8

###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/indep.py: Platform independent script with helper functions
# for CPT.
#
# Author: Anirudha Bose <ani07nov@gmail.com>
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#
###############################################################################

# Uncomment the following line to trace the execution of the shell commands
# set -o xtrace

# Import modules

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
from datetime import datetime, tzinfo
import multiprocessing
import fileinput

# Platform initialization
OS=platform.system()
DIST=platform.dist()[0]
PSEUDONAME=platform.dist()[2]
REV=platform.dist()[1]

# Extension initialization with default values
if OS == 'Windows':
    EXEEXT = '.exe'
    SHLIBEXT = '.dll'
elif OS == 'Linux':
    EXEEXT = ''
    SHLIBEXT = '.so'
elif OS == 'Darwin':
    EXEEXT = ''
    SHLIBEXT = '.dylib'
else:
    EXEEXT = ''
    SHLIBEXT = ''

# Global variable initialization
cmdline = ["cmd", "/q", "/k", "echo off"]
workdir = '/home/ani/ec/build'
srcdir = '/home/ani/ec/build/cling-src'
CLING_SRC_DIR = CLING_SRC_DIR + '/tools/cling'
LLVM_OBJ_ROOT = workdir + '/builddir'
LLVM_GIT_URL = 'http://root.cern.ch/git/llvm.git'
CLANG_GIT_URL = 'http://root.cern.ch/git/clang.git'
CLING_GIT_URL = 'http://root.cern.ch/git/cling.git'
LLVMRevision = urllib2.urlopen("https://raw.githubusercontent.com/ani07nov/cling/master/LastKnownGoodLLVMSVNRevision.txt").readline().rstrip()


def box_draw_header():
  msg='cling (' + platform.machine() + ')' + formatdate(float(datetime.now().strftime("%s")),tzinfo())
  spaces_no = 80 - len(msg) - 4
  spacer = ' ' * spaces_no
  msg='cling (' + platform.machine() + ')' + spacer + formatdate(float(datetime.now().strftime("%s")),tzinfo())
  print '''
╔══════════════════════════════════════════════════════════════════════════════╗
║ %s ║
╚══════════════════════════════════════════════════════════════════════════════╝'''%(msg)


def box_draw(msg):
  spaces_no = 80 - len(msg) - 4
  spacer = ' ' * spaces_no
  print '''
┌──────────────────────────────────────────────────────────────────────────────┐
│ %s%s │
└──────────────────────────────────────────────────────────────────────────────┘'''%(msg, spacer)


def fetch_llvm():
    box_draw("Fetch source files")
    print "Last known good LLVM revision is: %s"%(LLVMRevision)
    def get_fresh_llvm():
        subprocess.Popen(['git clone %s %s'%(LLVM_GIT_URL, srcdir)],
                         cwd=workdir,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git checkout ROOT-patches-r%s'%(LLVMRevision)],
                         cwd=srcdir,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    def update_old_llvm():
        subprocess.Popen(['git clean -f -x -d'],
                         cwd=srcdir,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git fetch --tags'],
                         cwd=srcdir,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git checkout ROOT-patches-r%s'%(LLVMRevision)],
                         cwd=srcdir,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git pull origin refs/tags/ROOT-patches-r%s'%(LLVMRevision)],
                         cwd=srcdir,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    if platform.system() == 'Windows':
        pass
    else:

        if os.path.isdir(srcdir):
            update_old_llvm()
        else:
            get_fresh_llvm()


def fetch_clang():
    def get_fresh_clang():
        subprocess.Popen(['git clone %s %s/tools/clang'%(CLANG_GIT_URL, srcdir)],
                         cwd=srcdir+'/tools',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git checkout ROOT-patches-r%s'%(LLVMRevision)],
                         cwd=srcdir+'tools/clang',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    def update_old_clang():
        subprocess.Popen(['git clean -f -x -d'],
                         cwd=srcdir+'/tools/clang',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git fetch --tags'],
                         cwd=srcdir+'/tools/clang',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git checkout ROOT-patches-r%s'%(LLVMRevision)],
                         cwd=srcdir+'/tools/clang',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git pull origin refs/tags/ROOT-patches-r%s'%(LLVMRevision)],
                         cwd=srcdir+'/tools/clang',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    if platform.system() == 'Windows':
        pass
    else:

        if os.path.isdir(srcdir+'/tools/clang'):
            update_old_clang()
        else:
            get_fresh_clang()


def set_version():
    box_draw("Set Cling version")
    VERSION=open(CLING_SRC_DIR+'/VERSION', 'r').readline().rstrip()

    # If development release, then add revision to the version
    REVISION=subprocess.Popen(['git log -n 1 --pretty=format:"%H"'],
                     cwd=CLING_SRC_DIR,
                     shell=True,
                     stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,
                     close_fds=True).communicate()[0].rstrip()

    if VERSION.find('~dev'):
        VERSION = VERSION + '-' + REVISION[:7]

    print 'Version: ' + VERSION
    print 'Revision: ' + REVISION


def set_ext():
    global EXEEXT
    global SHLIBEXT
    box_draw("Set binary/library extensions")
    if not os.path.isfile(LLVM_OBJ_ROOT + '/test/lit.site.cfg'):
        subprocess.Popen(['make lit.site.cfg'],
                         cwd=LLVM_OBJ_ROOT + '/test',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    with open(LLVM_OBJ_ROOT + '/test/lit.site.cfg', 'r') as lit_site_cfg:
        for line in lit_site_cfg:
            if re.match('^config.llvm_shlib_ext = ', line):
                SHLIBEXT = re.sub('^config.llvm_shlib_ext = ', '', line).replace('"', '').rstrip()
            elif re.match('^config.llvm_exe_ext = ', line):
                EXEEXT = re.sub('^config.llvm_exe_ext = ', '', line).replace('"', '').rstrip()

    print 'EXEEXT: ' + EXEEXT
    print 'SHLIBEXT: ' + SHLIBEXT


def compile(prefix):
    python=sys.executable
    cores=multiprocessing.cpu_count()

    # Cleanup previous installation directory if any
    shutil.rmtree(prefix)
    os.makedirs(workdir + '/builddir')

    if platform.system() == 'Windows':
        box_draw("Configure Cling with CMake and generate Visual Studio 11 project files")
        subprocess.Popen(['cmake -G "Visual Studio 11" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%s ../os.path.basename(%s)'%(TMP_PREFIX, srcdir)],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        box_draw("Building Cling (using %s cores)"%(cores))
        subprocess.Popen(['cmake --build . --target clang --config Release'],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['cmake --build . --target cling --config Release'],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        box_draw("Install compiled binaries to prefix (using %s cores)"%(cores))
        subprocess.Popen(['cmake --build . --target INSTALL --config Release'],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()
    else:
        box_draw("Configure Cling with GNU Make")
        subprocess.Popen(['%s/configure --disable-compiler-version-checks --with-python=%s --enable-targets=host --prefix=%s --enable-optimized=yes --enable-cxx11'%(srcdir, python, TMP_PREFIX)],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        box_draw("Building Cling (using %s cores)"%(cores))
        subprocess.Popen(['make -j%s'%(cores)],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        box_draw("Install compiled binaries to prefix (using %s cores)"%(cores))
        subprocess.Popen(['make install -j%s prefix=%s'%(cores, TMP_PREFIX)],
                         cwd=LLVM_OBJ_ROOT,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()


def install_prefix():
    set_ext()
    box_draw("Filtering Cling's libraries and binaries")
    print "This is going to take a while. Please wait."
    for line in fileinput.input(CLING_SRC_DIR + "/tools/packaging/dist-files.mk", inplace=True):
        if line.find("@EXEEXT@"):
            print line.replace("@EXEEXT@", EXEEXT),
        elif line.find("@SHLIBEXT@"):
            print line.replace("@SHLIBEXT@", SHLIBEXT),


    with open(CLING_SRC_DIR + '/tools/packaging/dist-files.mk', 'r') as dist-files:
        for root, dirs, files in os.walk(TMP_PREFIX):
            for file in files:
                f=os.path.join(root, file).replace(TMP_PREFIX, '')
                for line in dist-files:
                    if re.match('^\s\s%s\s'%(f), line):
                        os.makedirs(prefix + '/' + os.path.dirname(f))
                        shutil.copy(TMP_PREFIX + '/' + f, prefix + '/' f)

def test_cling():
    box_draw("Run Cling test suite")
    if platform.system() != 'Windows':
        subprocess.Popen(['make test'],
                         cwd=workdir + '/builddir/tools/cling',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()


def tarball():
    box_draw("Compressing binaries to produce a bzip2 tarball")
    tar = tarfile.open(prefix+'tar.bz2', 'w:bz2')
    tar.add(prefix, arcname=os.path.basename(prefix))
    t.close()


def cleanup():
    print "\n"
    box_draw("Clean up")
    if os.path.isdir(workdir+'/builddir'):
        print "Remove directory: " + workdir + "builddir"
        shutil.rmtree(workdir + '/buildir')

    if os.path.isdir(prefix):
        print "Remove directory: " + prefix
        shutil.rmtree(prefix)

    if os.path.isdir(TMP_PREFIX):
        print "Remove directory: " + TMP_PREFIX
        shutil.rmtree(TMP_PREFIX)

    if os.path.isfile(workdir + '/cling.nsi'):
        print "Remove file: cling.nsi"
        os.remove(workdir + '/cling.nsi')

    if VALUE == "deb" or PARAM == "--deb-tag":
        print 'Create output directory: %s/cling-%s-1'%(workdir, VERSION)
        os.makedirs('%s/cling-%s-1'%(workdir, VERSION))

        for file in glob.glob(r'%s/cling_%s*'%(workdir, VERSION)):
            print file + '->' + '%s/cling-%s-1'%(workdir, VERSION) + '/file'
            shutil.move(file, '%s/cling-%s-1'%(workdir, VERSION))

        if not os.listdir('%s/cling-%s-1'%(workdir, VERSION)):
            os.rmdir('%s/cling-%s-1'%(workdir, VERSION))
