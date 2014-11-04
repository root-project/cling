Cling Packaging Tool (CPT)
==========================

The Cling Packaging Tool is a command-line utility written in Python to build
Cling from source and generate installer bundles for a wide range of platforms.

Cling maintains its own vendor clones of LLVM and Clang (part of ROOT's trunk)
on which it is based. Due to some policy restrictions we do not distribute
Cling on official repositories of Debian and others. Therefore this tool is the
easiest way to build Cling for your favorite platorm and bundle it into an
installer. If you want to manually compile Cling from source, go through the
[README] of Cling or the build instructions [here].

[README]:https://github.com/vgvassilev/cling/blob/master/README.md
[here]:http://root.cern.ch/drupal/content/cling-build-instructions

Below is a list of platforms currently supported by CPT:
  * Ubuntu and distros based on Debian - *DEB packages*
  * Windows - *NSIS installers*
  * Distros based on Red Hat Linux (Fedora/Scientific Linux CERN) - *RPM packages*
  * Mac OS X - *Apple Disk Images*
  * Virtually any UNIX-like platform which supports Bash - *Tarballs*.

###Requirements
Before using this tool, make sure you have the required packages installed on
your system. Detailed information on what and how to install is provided below,
but the recommended (and much easier) way is to use the following command which
performs the required checks automatically and displays useful suggestions too
specific to your platform.
```sh
cd tools/packaging/
./cpt.py --check-requirements
```
or
```sh
cd tools/packaging/
./cpt.py -c
```
Regardless of the platform and operating system, make sure your system has the
latest and greatest version of Python 2 installed, v2.7 being the absolute minimum.
CPT uses some features and modules which are not a part of older versions of Python.
The same holds true for the versions of GCC/Clang you have on your machine. Older
compilers do not support c++11 features and thus you can expect a build error if you
choose not to update them.

All pre-compiled binaries of Python ship with built-in support for SSL. However if
the Python on your system was compiled by you manually, chances are that it doesn't
have SSL support. This is very likely if you had performed a minimal installation
of Scientific Linux CERN which doesn't include OpenSSL development package. In such
a case, you should install ```openssl-devel```, re-compile Python and ```configure```
will automatically link against the required libraries and produce a binary with SSL
support.

####Ubuntu/Debian
On Debian, Ubuntu, Linux Mint, CrunchBang, or any other distro based on Debian
which supports APT package manager, you can install all the required packages by:
```sh
sudo apt-get update
sudo apt-get install git g++ debhelper devscripts gnupg python
```
You are not required to do this manually since CPT can do this for you automatically.

######Setting up:
Make sure GnuPG is properly set up with your correct fingerprint. These
credentials are needed to sign the Debian package and create Debian changelogs.
On a build machine (Electric Commander), make sure the fingerprint is of the
user who is supposed to sign the official uploads. You might also want to
configure GnuPG to not ask for the passphrase while signing the Debian package.

The [Ubuntu Packaging Guide] contains a quick guide on creating a GPG key on an
Ubuntu system.

To test if you have successfully set up your GnuPG key, use the following command:
```sh
gpg --fingerprint
```
Again, all these checks are performed by default when you launch CPT with ```-c``` option.
[Ubuntu Packaging Guide]:http://packaging.ubuntu.com/html/getting-set-up.html#create-your-gpg-key

####Windows
CPT is meant to be executed on cmd.exe prompt. Make sure you have set the
environment properly before continuing.
Below is a list of required packages for Windows (Win32-x86):

[MSYS Git] for Windows

[Python] for Windows

Microsoft Visual Studio 11 (2012), with Microsoft Visual C++ 2012

[MSYS Git]:http://msysgit.github.io/
[Python]:https://www.python.org/

######Setting Up:
Unlike other UNIX-like platforms, Windows requires you to follow some rules.
Do not ignore this section unless you want CPT to fail mid-way with wierd
errors. You should require these instructions only once.

  * While installing the packages make sure the executable is in a path that
doesn't contain spaces. For example, you should install Python in a path like

    ```sh
    C:\Python27
    ```
    rather than

    ```sh
    C:\Program Files (x86)\Python 2.7
    ```
  * Path to all the required executables should be present in the Windows
    **PATH** environment variable.
  * In case of MSYS Git, choose the option "Run Git from Windows
    Command Prompt" during installation.

A good way to check if everything is detected properly by the script is to
run the following command:
```sh
cd tools/packaging/
./cpt.py --check-requirements
```

####Red Hat Linux (Fedora/Scientific Linux CERN)
This section applies to all distros based on Red Hat Linux like Fedora, and
Scientific Linux CERN (SLC). Apparently, you can build RPM packages in any
distro regardless of the package manager it uses. This has been tested on
Fedora, SLC, Ubuntu, and CrunchBang. If you are interested, you can test it
on your favourite platform and email me the results.

Depending on the package manager of your distro, you can install the
packages required by CPT to build RPM bundles. For a Red Hat based distro
(which uses ```yum``` package manager), you can use the following command
(also performed automatically by CPT):
```sh
sudo yum update
sudo yum install git gcc gcc-c++ rpm-build python
```

####Mac OS X
Mac OS X provides a sane environement for CPT to build Apple Disk Images
(DMG Installers). On older versions of Mac OS, you need to update XCode to
get the latest version of Clang supporting c++11 features. A great package
manager for Mac OS X is [Macports]. It is recommended that you use the
packages provided by Macports for running CPT (or any other tool if that
is the case) rather than the ones which come pre-installed with Mac OS.
Assuming that you have Macports installed on your Mac, you can use the
following command to install the requisite packages (also done automatically
by CPT):
[Macports]:http://www.macports.org/
```sh
sudo port -v selfupdate
sudo port install git g++ python
```

###Usage
```sh
cd tools/packaging/
```

```
usage: cpt.py [-h] [-c] [--current-dev CURRENT_DEV]
              [--last-stable LAST_STABLE] [--tarball-tag TARBALL_TAG]
              [--deb-tag DEB_TAG] [--rpm-tag RPM_TAG] [--nsis-tag NSIS_TAG]
              [--dmg-tag DMG_TAG] [--with-llvm-url WITH_LLVM_URL]
              [--with-clang-url WITH_CLANG_URL]
              [--with-cling-url WITH_CLING_URL] [--no-test]
              [--create-dev-env CREATE_DEV_ENV] [--with-workdir WITH_WORKDIR]
              [--make-proper MAKE_PROPER]

Cling Packaging Tool

optional arguments:
  -h, --help            show this help message and exit
  -c, --check-requirements
                        Check if packages required by the script are installed
  --current-dev CURRENT_DEV
                        Package the latest development snapshot in one of
                        these formats: tar | deb | nsis | rpm | dmg | pkg
  --last-stable LAST_STABLE
                        Package the last stable snapshot in one of these
                        formats: tar | deb | nsis | rpm | dmg | pkg
  --tarball-tag TARBALL_TAG
                        Package the snapshot of a given tag in a tarball
                        (.tar.bz2)
  --deb-tag DEB_TAG     Package the snapshot of a given tag in a Debian
                        package (.deb)
  --rpm-tag RPM_TAG     Package the snapshot of a given tag in an RPM package
                        (.rpm)
  --nsis-tag NSIS_TAG   Package the snapshot of a given tag in an NSIS
                        installer (.exe)
  --dmg-tag DMG_TAG     Package the snapshot of a given tag in a DMG package
                        (.dmg)
  --with-llvm-url WITH_LLVM_URL
                        Specify an alternate URL of LLVM repo
  --with-clang-url WITH_CLANG_URL
                        Specify an alternate URL of Clang repo
  --with-cling-url WITH_CLING_URL
                        Specify an alternate URL of Cling repo
  --no-test             Do not run test suite of Cling
  --create-dev-env CREATE_DEV_ENV
                        Set up a release/debug environment
  --with-workdir WITH_WORKDIR
                        Specify an alternate working directory for CPT
  --make-proper MAKE_PROPER
                        Internal option to support calls from build system

```
If you want CPT to build a package by detecting your platform automatically,
use the value 'pkg'.
```sh
./cpt.py --current-dev=pkg
```
or
```sh
./cpt.py --last-stable=pkg
```
###Overriding Default Variables
There are a select number of variables which can be set to make CPT work
differently. This eliminates the need to manually edit the script.
You can overrride variables by using the following syntax:
```$ ./cpt.py --with-cling-url="http://github.com/ani07nov/cling" --current-dev=tar```.

List of variables in CPT which can be overridden:
- **CLING_GIT_URL**
  * Specify the URL of the Git repository of Cling to be used by CPT
  * **Default value:** "http://root.cern.ch/git/cling.git"
  * **Usage:** ```./cpt.py --with-cling-url="http://github.com/ani07nov/cling" --last-stable=deb```

- **CLANG_GIT_URL**
  * Specify the URL of the Git repository of Clang to be used by CPT
  * **Default value:** "http://root.cern.ch/git/clang.git"
  * **Usage:** ```./cpt.py --with-clang-url="http://github.com/ani07nov/clang" --last-stable=tar```

- **LLVM_GIT_URL**
  * Specify the URL of the Git repository of LLVM to be used by CPT
  * **Default value:** "http://root.cern.ch/git/llvm.git"
  * **Usage:** ```./cpt.py --with-llvm-url="http://github.com/ani07nov/llvm" --current-dev=tar```

- **workdir**
  * Specify the working directory of CPT. All sources will be cloned, built
    and installed here. The produced packages will also be found here.
  * **Default value:** "~/ec/build"
  * **Usage:** ```./cpt.py --with-workdir="/ec/build" --current-dev=deb```

Authors
=======
Cling Packaging Tool was developed during Google Summer of Code 2014,
by Anirudha Bose under the mentorship of Vassil Vassilev.

Please post all bug reports and feature requests in the Github issue tracker
of this repository. Alternatively you can directly email me in this address:
ani07nov@gmail.com

License
=======
Cling Packaging Tool is a part of Cling project and released under the same
license terms of Cling. You can choose to license it under the University of
Illinois Open Source License or the GNU Lesser General Public License. See
[LICENSE.TXT] for details.

[LICENSE.TXT]:https://github.com/vgvassilev/cling/blob/master/LICENSE.TXT
