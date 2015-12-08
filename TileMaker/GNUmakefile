#
# This is an example GNUmakefile for my packages
#

# specific names for this package
DICT  = UbooneLiveBackEnd_TileMakerCint
SHLIB = libUbooneLiveBackEnd_TileMaker.so
SOURCES = $(filter-out $(DICT).cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h $(DICT).h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
OBJECTS = $(SOURCES:.cxx=.o)

CXXFLAGS += -DNO_FREETYPE

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell larlite-config --includes)
INCFLAGS += $(shell python-config --includes)
INCFLAGS += -I$(shell python -c "import numpy; print numpy.get_include()")
INCFLAGS += -I..

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

# LDFLAGS += $(shell larlite-config --libs) -lDisplayTool_RawViewer

CINTFLAGS += "-p"
CINTFLAGS += "-DNO_FREETYPE"
# call kernel specific compiler setup
include $(LARLITE_BASEDIR)/Makefile/Makefile.${OSNAME}
# call the common GNUmakefile
include $(LARLITE_BASEDIR)/Makefile/GNUmakefile.CORE
