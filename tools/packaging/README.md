Cling Packaging Tool (CPT)
==========================

The Cling Packaging Tool is a command-line utility to build Cling from source
and generate installer bundles for a wide range of platforms.

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
  * Cygwin or Windows - *NSIS installers*
  * Virtually any UNIX-like platform which supports Bash - *Tarballs*.

###Requirements
Before using this tool, make sure you have the required packages installed on
your system. Detailed information on what and how to install is provided below,
but the recommended (and much easier) way is to use the following command which
performs the required checks automatically and displays useful suggestions too
specific to your platform.
```sh
cd tools/packaging/
./cpt.sh --check-requirements
```

####Ubuntu/Debian
On Debian, Ubuntu or any other distro based on Debian which supports APT
package manager, you can install all the required packages by:
```sh
sudo apt-get update
sudo apt-get install git wget debhelper devscripts gnupg python
```

######Setting up:
Make sure GnuPG is properly set up with your correct fingerprint. These
credentials are needed to sign the Debian package and create Debian changelogs.
On a build machine (Electric Commander), make sure the fingerprint is of the
person who is supposed to sign the official uploads. On a build machine, you
might also want to configure GnuPG to not ask for the passphrase while signing
the Debian package.

####Windows (Cygwin)
Below is a list of required packages for Windows (Win32-x86):

[CMake] for Windows  
[MSYS Git] or Git provided by Cygwin  
[Cygwin]  
[Python] for Windows  
wget - provided by Cygwin  
Microsoft Visual Studio 11 (2012), with Microsoft Visual C++ 2012
[CMake]:http://www.cmake.org/
[MSYS Git]:http://msysgit.github.io/
[Cygwin]:http://www.cygwin.com/
[Python]:https://www.python.org/

**Note:** Git provided by Cygwin is recommended over MSYS Git due to some
known issues.
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
  * If you plan to use MSYS Git, choose the option "Run Git from Windows
    Command Prompt" during installation.
  * Add these lines in the file ~/.bash_profile of Cygwin. UNIX files just
    don't run on Windows without this tweak.

    ```sh
    export SHELLOPTS
    set -o igncr
    ```

A good way to check if everything is detected properly by the script is to
run the following command:
```sh
cd tools/packaging/
./cpt.sh --check-requirements
```
**Tip:** To make things easy for yourself in future, you should keep the Cygwin
installer file you had downloaded previously in a safe place. Cygwin allows an
easy way to install new packages through command-line. See an example below:
```sh
/cygdrive/c/cygwin/setup-x86.exe -nqP wget
```

License
----
Cling Packaging Tool is a part of Cling project and released under the same
license terms of Cling. You can choose to license it under the University of
Illinois Open Source License or the GNU Lesser General Public License. See
[LICENSE.TXT] for details.

[LICENSE.TXT]:https://github.com/vgvassilev/cling/blob/master/LICENSE.TXT
