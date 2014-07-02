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
from datetime import datetime, tzinfo
import multiprocessing
import fileinput

###############################################################################
#                           Platform initialization                           #
###############################################################################

OS=platform.system()
DIST=platform.dist()[0]
RELEASE=platform.dist()[2]
REV=platform.dist()[1]

if OS == 'Windows':
    EXEEXT = '.exe'
    SHLIBEXT = '.dll'
elif OS == 'Linux':
    EXEEXT = ''
    SHLIBEXT = '.so'

    TMP_PREFIX='/var/tmp/cling-obj/'
elif OS == 'Darwin':
    EXEEXT = ''
    SHLIBEXT = '.dylib'
else:
    EXEEXT = ''
    SHLIBEXT = ''

###############################################################################
#                               Global variables                              #
###############################################################################

cmdline = ["cmd", "/q", "/k", "echo off"]
workdir = '/home/ani/ec/build'
srcdir = '/home/ani/ec/build/cling-src'
prefix = ''
CLING_SRC_DIR = srcdir + '/tools/cling'
LLVM_OBJ_ROOT = workdir + '/builddir'
LLVM_GIT_URL = 'http://root.cern.ch/git/llvm.git'
CLANG_GIT_URL = 'http://root.cern.ch/git/clang.git'
CLING_GIT_URL = 'http://root.cern.ch/git/cling.git'
LLVMRevision = urllib2.urlopen("https://raw.githubusercontent.com/ani07nov/cling/master/LastKnownGoodLLVMSVNRevision.txt").readline().rstrip()
VERSION=''


###############################################################################
#              Platform independent functions (formerly indep.py)             #
###############################################################################

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

def fetch_cling(arg):
    def get_fresh_cling():
        subprocess.Popen(['git clone %s %s'%(CLING_GIT_URL, CLING_SRC_DIR)],
                         cwd=srcdir+'/tools',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()
        if arg == 'last-stable':
            checkout_branch = subprocess.Popen(['git describe --match v* --abbrev=0 --tags | head -n 1'],
                                               cwd=CLING_SRC_DIR,
                                               shell=True,
                                               stdin=subprocess.PIPE,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT,
                                               close_fds=True).communicate()[0]
        elif arg == 'master':
            checkout_branch = 'master'
        else:
            checkout_branch = arg

        subprocess.Popen(['git checkout %s'%(checkout_branch)],
                         cwd=CLING_SRC_DIR,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    def update_old_cling():
        subprocess.Popen(['git clean -f -x -d'],
                         cwd=CLING_SRC_DIR,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git fetch --tags'],
                         cwd=CLING_SRC_DIR,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        if arg == 'last-stable':
            checkout_branch = subprocess.Popen(['git describe --match v* --abbrev=0 --tags | head -n 1'],
                                               cwd=CLING_SRC_DIR,
                                               shell=True,
                                               stdin=subprocess.PIPE,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT,
                                               close_fds=True).communicate()[0]
        elif arg == 'master':
            checkout_branch = 'master'
        else:
            checkout_branch = arg

        subprocess.Popen(['git checkout %s'%(checkout_branch)],
                         cwd=CLING_SRC_DIR,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

        subprocess.Popen(['git pull origin %s'%(checkout_branch)],
                         cwd=CLING_SRC_DIR,
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()

    if platform.system() == 'Windows':
        pass
    else:
        if os.path.isdir(CLING_SRC_DIR):
            update_old_cling()
        else:
            get_fresh_cling()


def set_version():
    global VERSION
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

    if '~dev' in VERSION:
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
    if os.path.isdir(workdir + '/builddir'):
        print "Remove directory: " + workdir + '/builddir'
        shutil.rmtree(workdir + '/builddir')

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

    for line in fileinput.input(CLING_SRC_DIR + '/tools/packaging/dist-files.mk', inplace=True):
        if '@EXEEXT@' in line:
            print line.replace('@EXEEXT@', EXEEXT),
        elif '@SHLIBEXT@' in line:
            print line.replace('@SHLIBEXT@', SHLIBEXT),
        else:
            print line,

    dist_files = open(CLING_SRC_DIR + '/tools/packaging/dist-files.mk', 'r').read()
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
        subprocess.Popen(['make test'],
                         cwd=workdir + '/builddir/tools/cling',
                         shell=True,
                         stdin=subprocess.PIPE,
                         stdout=None,
                         stderr=subprocess.STDOUT,
                         close_fds=True).communicate()


def tarball():
    box_draw("Compress binaries into a bzip2 tarball")
    tar = tarfile.open(prefix+'.tar.bz2', 'w:bz2')
    tar.add(prefix, arcname=os.path.basename(prefix))
    tar.close()


def cleanup():
    print "\n"
    box_draw("Clean up")
    if os.path.isdir(os.path.join(workdir,'builddir')):
        print "Remove directory: " + os.path.join(workdir,'builddir')
        shutil.rmtree(os.path.join(workdir,'builddir'))

    if os.path.isdir(prefix):
        print "Remove directory: " + prefix
        shutil.rmtree(prefix)

    if os.path.isdir(TMP_PREFIX):
        print "Remove directory: " + TMP_PREFIX
        shutil.rmtree(TMP_PREFIX)

    if os.path.isfile(os.path.join(workdir,'cling.nsi')):
        print "Remove file: " + os.path.join(workdir,'cling.nsi')
        os.remove(os.path.join(workdir,'cling.nsi'))

    if args['current_dev'] == 'deb' or args['last-stable'] == 'deb' or args['deb-tag']:
        print 'Create output directory: %s/cling-%s-1'%(workdir, VERSION)
        os.makedirs('%s/cling-%s-1'%(workdir, VERSION))

        for file in glob.glob(r'%s/cling_%s*'%(workdir, VERSION)):
            print file + '->' + '%s/cling-%s-1'%(workdir, VERSION)
            shutil.move(file, '%s/cling-%s-1'%(workdir, VERSION))

        if not os.listdir('%s/cling-%s-1'%(workdir, VERSION)):
            os.rmdir('%s/cling-%s-1'%(workdir, VERSION))


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
parser.add_argument('--with-workdir', help='Specify an alternate working directory for CPT', default=os.path.expanduser('~/ec/build'))

args = vars(parser.parse_args())


print 'Cling Packaging Tool (CPT)'
print 'Arguments vector: ' + str(sys.argv)
box_draw_header()
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
        compile(workdir + '/cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION)
        install_prefix()
        test_cling()
        tarball()
        cleanup()

    elif args['current_dev'] == "deb":
        compile(workdir + '/cling-' + VERSION)
        install_prefix()
        test_cling()
        #tarball_deb()
        #debianize()
        cleanup()
    elif args['current_dev'] == 'nsis':
        compile(workdir + '/cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION)
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
        compile(workdir + '/cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION)
        install_prefix()
        test_cling()
        tarball()
        cleanup()
    if args['last_stable'] == 'deb':
        set_version()
        compile(workdir + '/cling-' + VERSION)
        install_prefix()
        test_cling()
        #tarball_deb()
        #debianize()
        cleanup()
    if args['last_stable'] == 'nsis':
        compile(workdir + '/cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION)
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
    compile(workdir + '/cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION)
    install_prefix()
    test_cling()
    tarball()
    cleanup()

if args['deb_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['deb_tag'])
    set_version()
    compile(workdir + '/cling-' + VERSION)
    install_prefix()
    test_cling()
    #tarball_deb
    #debianize
    cleanup()

if args['nsis_tag']:
    fetch_llvm()
    fetch_clang()
    fetch_cling(args['nsis_tag'])
    set_version()
    compile(workdir + '/cling-' + DIST + '-' + REV + '-' + platform.machine() + '-' + VERSION)
    install_prefix()
    test_cling()
    #get_nsis
    #make_nsi
    #build_nsis
    cleanup()
