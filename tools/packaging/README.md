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

Below is a list of platforms currently supported by this tool:
  * Ubuntu and distros based on Debian - *DEB packages*
  * Windows - *NSIS installers*
  * Distros based on Red Hat Linux (Fedora/Scientific Linux CERN) - *RPM packages*
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

####Ubuntu/Debian
On Debian, Ubuntu or any other distro based on Debian which supports APT
package manager, you can install all the required packages by:
```sh
sudo apt-get update
sudo apt-get install git g++ debhelper devscripts gnupg python
```

######Setting up:
Make sure GnuPG is properly set up with your correct fingerprint. These
credentials are needed to sign the Debian package and create Debian changelogs.
On a build machine (Electric Commander), make sure the fingerprint is of the
person who is supposed to sign the official uploads. You might also want to
configure GnuPG to not ask for the passphrase while signing the Debian package.

The [Ubuntu Packaging Guide] contains documentation about creating a GPG key
on an Ubuntu system.

To test if you have successfully set up your GnuPG key, use the following command:
```sh
gpg --fingerprint
```

Again, all these checks are performed by default when you launch CPT with -c option.
```sh
./cpt.py -c
```
[Ubuntu Packaging Guide]:http://packaging.ubuntu.com/html/getting-set-up.html#create-your-gpg-key

####Windows
CPT is meant to be executed on cmd.exe prompt. Make sure you have set the
environment properly before continuing.
Below is a list of required packages for Windows (Win32-x86):

[MSYS Git]  
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
###Usage
```sh
cd tools/packaging/
```

```
usage: cpt.py [-h] [-c] [--current-dev CURRENT_DEV]
              [--last-stable LAST_STABLE] [--tarball-tag TARBALL_TAG]
              [--deb-tag DEB_TAG] [--rpm-tag RPM_TAG] [--nsis-tag NSIS_TAG]
              [--with-llvm-url WITH_LLVM_URL]
              [--with-clang-url WITH_CLANG_URL]
              [--with-cling-url WITH_CLING_URL] [--with-workdir WITH_WORKDIR]

Cling Packaging Tool

optional arguments:
  -h, --help            show this help message and exit
  -c, --check-requirements
                        Check if packages required by the script are installed
  --current-dev CURRENT_DEV
                        Package the latest development snapshot in one of
                        these formats: tar | deb | nsis
  --last-stable LAST_STABLE
                        Package the last stable snapshot in one of these
                        formats: tar | deb | nsis
  --tarball-tag TARBALL_TAG
                        Package the snapshot of a given tag in a tarball
                        (.tar.bz2)
  --deb-tag DEB_TAG     Package the snapshot of a given tag in a Debian
                        package (.deb)
  --rpm-tag RPM_TAG     Package the snapshot of a given tag in an RPM package
                        (.rpm)
  --nsis-tag NSIS_TAG   Package the snapshot of a given tag in an NSIS
                        installer (.exe)
  --with-llvm-url WITH_LLVM_URL
                        Specify an alternate URL of LLVM repo
  --with-clang-url WITH_CLANG_URL
                        Specify an alternate URL of Clang repo
  --with-cling-url WITH_CLING_URL
                        Specify an alternate URL of Cling repo
  --with-workdir WITH_WORKDIR
                        Specify an alternate working directory for CPT
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
  * **Usage:** ```./cpt.py --with-workdir="/ec/build/cling" --current-dev=deb```

License
=======
Cling Packaging Tool is a part of Cling project and released under the same
license terms of Cling. You can choose to license it under the University of
Illinois Open Source License or the GNU Lesser General Public License. See
[LICENSE.TXT] for details.

[LICENSE.TXT]:https://github.com/vgvassilev/cling/blob/master/LICENSE.TXT
