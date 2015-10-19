SHELL := /bin/bash
 
ifndef config
	config=release
endif

BUILDROOT := ../build
BUILDDIR  := $(BUILDROOT)/$(config)
BINDIR    := $(BUILDDIR)/bin
OBJDIR    := $(BUILDDIR)/obj
CPPDIR    := .
 
NVCC = nvcc
CC = g++
LD = g++
CFLAGS    := -m64 -fPIC -fvisibility=hidden \
             --param max-inline-recursive-depth=24 --param max-inline-recursive-depth-auto=24
CXXFLAGS  := 
CXX11FLAGS:= -std=gnu++11
NVCCFLAGS := -use_fast_math -gencode arch=compute_35,code=sm_35 -include nvcc_preinclude.h -maxrregcount 64

LDFLAGS   := -shared -shared-libgcc -Bsymbolic -fvisibility=hidden -Wl,-rpath,'$$ORIGIN' -Wl,--no-undefined -Wl,--export-dynamic
DEFINES   := 
INCLUDE   := -I$(CPPDIR)
LIBPATH   := -L$(BINDIR)
LIBPREFIX := lib
LIBSUFFIX := .so

ifndef CUDADIR
	CUDADIR   := /usr/local/cuda
endif

ifndef SMATDIR
	SMATDIR   := ../../smat
endif

ifndef prefix
	prefix    := ../install
endif

ifeq ($(config),debug)
	CFLAGS += -g3
else
	CFLAGS += -O3
	LDFLAGS += -Xlinker --strip-debug
endif

all: kangaroo_smat

clean:
	@rm -fr $(BUILDROOT)/debug/*
	@rm -fr $(BUILDROOT)/release/*

CUDA_INCLUDE := -I$(CUDADIR)/include
CUDA_LIBPATH := -L$(CUDADIR)/lib64
CUDA_LDFLAGS := -Wl,-rpath,$(CUDADIR)/lib64

SMAT_INCLUDE := -I$(SMATDIR)/src
SMAT_LIBPATH := -L$(SMATDIR)/build/$(config)/bin
SMAT_LDFLAGS := -Wl,-rpath,$(SMATDIR)/build/$(config)/bin

######################### libsmat_ext_kangaroo.so ###########################

KANGAROO_SMAT_DEFINES := -DKANGAROO_SMAT_EXPORTS
KANGAROO_SMAT_INCLUDE := $(CUDA_INCLUDE) $(SMAT_INCLUDE)
KANGAROO_SMAT_LIBPATH := $(CUDA_LIBPATH) $(SMAT_LIBPATH)
KANGAROO_SMAT_LDFLAGS := $(CUDA_LDFLAGS) $(SMAT_LDFLAGS)
KANGAROO_SMAT_DIR := .
KANGAROO_SMAT_CPP := $(foreach dir,$(KANGAROO_SMAT_DIR),$(wildcard $(dir)/*.cpp))
KANGAROO_SMAT_CU  := $(foreach dir,$(KANGAROO_SMAT_DIR),$(wildcard $(dir)/*.cu))
KANGAROO_SMAT_LIB := base smat smat_cuda cuda cudart
KANGAROO_SMAT_DLL := $(BINDIR)/$(LIBPREFIX)kangaroo_smat$(LIBSUFFIX)

$(KANGAROO_SMAT_DLL): 
kangaroo_smat: $(KANGAROO_SMAT_DLL)


####################################################################

# Tell GNU make that these are not actual file targets
.PHONY: kangaroo_smat
.PHONY: all
.PHONY: clean

####################################################################

dummy=
space=$(dummy) $(dummy)
comma=$(dummy),$(dummy)

define join-list
$(subst $(space),$(2),$(1))
endef

####################################################################

# These are the rules that compile src/.../%.cpp into both
# object file build/config/.../%.o and dependency file
# build/config/.../%.d

define run-dep
@cp $(basename $@).d $(basename $@).d.tmp; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' -e '/^$$/ d'\
	    -e 's/$$/ :/' < $(basename $@).d >> $(basename $@).d.tmp;
@mv -f $(basename $@).d.tmp $(basename $@).d
endef

define run-cc
@mkdir -p $(@D)
$(CC) -MMD -c -o $@ $(INCLUDE) $($(1)_INCLUDE) \
                    $(DEFINES) $($(1)_DEFINES) \
                    $(CFLAGS) $(CXXFLAGS) $(CXX11FLAGS) $^
$(run-dep)
endef

define run-nvcc
@mkdir -p $(@D)
$(NVCC) -c -o $@ $(INCLUDE) $($(1)_INCLUDE) \
                 $(DEFINES) $($(1)_DEFINES) \
                 $(NVCCFLAGS) -Xcompiler $(call join-list,$(CFLAGS) $(CXXFLAGS),$(comma)) $^
$(NVCC) -M $(INCLUDE) $($(1)_INCLUDE) \
           $(DEFINES) $($(1)_DEFINES) \
           $(NVCCFLAGS) -Xcompiler $(call join-list,$(CFLAGS) $(CXXFLAGS),$(comma)) $^ > $(basename $@).d
$(run-dep)
endef

define gen-obj-rule-cpp
$(OBJDIR)/$(1)/$(basename $(notdir $(2))).o : $(2)
	$$(call run-cc,$(1),$(2))
endef

define gen-obj-rule-cu
$(OBJDIR)/$(1)/$(basename $(notdir $(2))).o : $(2)
	$$(call run-nvcc,$(1),$(2))
endef

define gen-module-rules
# Generate rules for .cpp and/or .cu with flags specifically for 
# the module $(1) (e.g. BASE,SMAT,SMAT_CUDA,etc)

$$(foreach file,$$($(1)_CPP),$$(eval $$(call gen-obj-rule-cpp,$(1),$$(file))))
$$(foreach file,$$($(1)_CU),$$(eval $$(call gen-obj-rule-cu,$(1),$$(file))))

$(1)_OBJ  = $$(foreach file,$$(patsubst %.cpp,%.o,$$(notdir $$($(1)_CPP))),$$(OBJDIR)/$(1)/$$(file))
$(1)_OBJ += $$(foreach file,$$(patsubst %.cu,%.o,$$(notdir $$($(1)_CU))),$$(OBJDIR)/$(1)/$$(file))

# Generate rule for linking 
$$($(1)_DLL): $$($(1)_OBJ)
	@mkdir -p $$(BINDIR)
	$$(LD) -o $$($(1)_DLL) $$($(1)_OBJ) \
	          $$(LDFLAGS) $$($(1)_LDFLAGS) \
	          $$(LIBPATH) $$($(1)_LIBPATH) \
	          $$(foreach lib,$$($(1)_LIB),-l$$(lib))

endef

#OBJECTS   := 

$(eval $(call gen-module-rules,KANGAROO_SMAT))

###########
# Auto-generation of header dependency rules 
# doesn't work yet, needs debugging 
#
# DEPENDS = $(patsubst %.o,%.d,$(OBJECTS))
# -include $(DEPENDS) /dev/null













