##===- docs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The Cling Interpreter
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

CLING_LEVEL := ..
PROJ_OBJ_DIR = .
DIRS       := tools

include $(CLING_LEVEL)/Makefile

DOXYGEN = doxygen

PROJ_docsdir := $(PROJ_docsdir)/cling

HTML       := $(wildcard $(PROJ_SRC_DIR)/doxygen/html/*.html) \
              $(wildcard $(PROJ_SRC_DIR)/*.css)
#IMAGES     := $(wildcard $(PROJ_SRC_DIR)/img/*.*)
DOXYFILES  := doxygen.cfg.in doxygen.css doxygen.footer doxygen.header \
              doxygen.intro
EXTRA_DIST := $(HTML) $(DOXYFILES) llvm.css CommandGuide img

.PHONY: install-html install-doxygen doxygen generated

install_targets :=
ifndef ONLY_MAN_DOCS
install_targets += install-html
endif
ifeq ($(ENABLE_DOXYGEN),1)
install_targets += install-doxygen
endif
install-local:: $(install_targets)

# Live documentation is generated for the web site using this target:
# 'make generated BUILD_FOR_WEBSITE=1'
generated:: doxygen

install-html: $(PROJ_OBJ_DIR)/html.tar.gz
	$(Verb) if test -d $(PROJ_OBJ_DIR)/doxygen ; then \
	  echo Installing HTML documentation ;\
	  $(MKDIR) $(DESTDIR)$(PROJ_docsdir)/html; \
	  $(MKDIR) $(DESTDIR)$(PROJ_docsdir)/html/img; \
	  $(DataInstall) $(HTML) $(DESTDIR)$(PROJ_docsdir)/html; \
	  $(DataInstall) $(PROJ_OBJ_DIR)/html.tar.gz $(DESTDIR)$(PROJ_docsdir); \
	fi

$(PROJ_OBJ_DIR)/html.tar.gz: $(HTML)
	$(Verb) if test -d $(PROJ_OBJ_DIR)/doxygen ; then \
	  echo Packaging HTML documentation; \
	  $(RM) -rf $@ $(PROJ_OBJ_DIR)/html.tar; \
	  cd $(PROJ_OBJ_DIR) && \
	    $(TAR) cf html.tar doxygen/html/; \
	  $(GZIPBIN) $(PROJ_OBJ_DIR)/html.tar; \
	fi

install-doxygen: doxygen
	$(Echo) Installing doxygen documentation
	$(Verb) $(MKDIR) $(DESTDIR)$(PROJ_docsdir)/html/doxygen
	$(Verb) $(DataInstall) $(PROJ_OBJ_DIR)/doxygen.tar.gz $(DESTDIR)$(PROJ_docsdir)
	$(Verb) cd $(PROJ_OBJ_DIR)/doxygen && \
	  $(FIND) . -type f -exec \
	    $(DataInstall) {} $(DESTDIR)$(PROJ_docsdir)/html/doxygen \;

doxygen: regendoc $(PROJ_OBJ_DIR)/doxygen.tar.gz

regendoc: gen-doxygen-cfg
	$(Echo) Building doxygen documentation
	$(Verb) if test -e $(PROJ_OBJ_DIR)/doxygen ; then \
	  $(RM) -rf $(PROJ_OBJ_DIR)/doxygen ; \
	fi
	$(Verb) $(DOXYGEN) $(PROJ_OBJ_DIR)/doxygen.cfg
	$(Verb) sed -i "s/[$$]LatestRev[$$]/`git show -s --pretty=format:%h $(PROJ_SRC_DIR)`/g" \
	 $(PROJ_OBJ_DIR)/doxygen/html/*.html

$(PROJ_OBJ_DIR)/doxygen.tar.gz: $(DOXYFILES) $(PROJ_OBJ_DIR)/doxygen.cfg
	$(Echo) Packaging doxygen documentation
	$(Verb) $(RM) -rf $@ $(PROJ_OBJ_DIR)/doxygen.tar
	$(Verb) $(TAR) cf $(PROJ_OBJ_DIR)/doxygen.tar doxygen
	$(Verb) $(GZIPBIN) $(PROJ_OBJ_DIR)/doxygen.tar
	$(Verb) $(CP) $(PROJ_OBJ_DIR)/doxygen.tar.gz $(PROJ_OBJ_DIR)/doxygen/html/

gen-doxygen-cfg:
	$(Echo) Generating doxygen configuration file
	$(Echo) $(PROJ_OBJ_DIR)/doxygen.cfg
	cat $(PROJ_SRC_DIR)/doxygen.cfg.in | sed \
	  -e 's,@docs_srcdir@,'$(PROJ_SRC_DIR)",g" \
	  -e 's,@cling_srcdir@,'$(call realpath, $(PROJ_SRC_DIR)/$(CLING_LEVEL))",g" \
	  -e 's/@DOT@/dot/g' \
	  -e 's/@PACKAGE_VERSION@/mainline/' \
	  -e 's/@docs_builddir@/./g' > $(PROJ_OBJ_DIR)/doxygen.cfg

userloc: $(LLVM_SRC_ROOT)/docs/userloc.html

$(LLVM_SRC_ROOT)/docs/userloc.html:
	$(Echo) Making User LOC Table
	$(Verb) cd $(LLVM_SRC_ROOT) ; ./utils/userloc.pl -details -recurse \
	  -html lib include tools runtime utils examples autoconf test > docs/userloc.html

uninstall-local::
	$(Echo) Uninstalling Documentation
	$(Verb) $(RM) -rf $(DESTDIR)$(PROJ_docsdir)
