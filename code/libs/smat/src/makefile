SHELL := /bin/bash
 
ifndef config
	config=release
endif

ifndef CUDA_ARCH
	CUDA_ARCH=30
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
NVCCFLAGS := -use_fast_math -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -include nvcc_preinclude.h

LDFLAGS   := -shared -shared-libgcc -Bsymbolic -fvisibility=hidden -fvisibility-inlines-hidden -Wl,-rpath,'$$ORIGIN' -Wl,--no-undefined -Wl,--export-dynamic
DEFINES   := 
INCLUDE   := -I$(CPPDIR)
LIBPATH   := -L$(BINDIR)
LIBPREFIX := lib
LIBSUFFIX := .so

ifndef CUDADIR
	CUDADIR   := /usr/local/cuda
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

all: base smat smat_cuda smat_ext_demo

clean:
	@rm -fr $(BUILDROOT)/debug/*
	@rm -fr $(BUILDROOT)/release/*

######################### libbase.so ################################

BASE_DEFINES := -DBASE_EXPORTS
BASE_INCLUDE := 
BASE_LIBPATH :=
BASE_LDFLAGS := -ldl
BASE_DIR := base
BASE_CPP := $(wildcard $(BASE_DIR)/*.cpp)
BASE_LIB := 
BASE_DLL := $(BINDIR)/$(LIBPREFIX)base$(LIBSUFFIX)

base: $(BASE_DLL)

######################### libsmat.so ################################

SMAT_DEFINES := -DSMAT_EXPORTS
SMAT_INCLUDE := 
SMAT_LIBPATH := 
SMAT_LDFLAGS := 
SMAT_DIR := smat smat/vm smat/vm/util
SMAT_CPP := $(foreach dir,$(SMAT_DIR),$(wildcard $(dir)/*.cpp))
SMAT_LIB := base
SMAT_DLL := $(BINDIR)/$(LIBPREFIX)smat$(LIBSUFFIX)

$(SMAT_DLL): $(BASE_DLL)
smat: $(SMAT_DLL)

######################### libsmat_cuda.so ###########################

SMAT_CUDA_DEFINES := -DSMAT_CUDA_EXPORTS
SMAT_CUDA_INCLUDE := -I$(CUDADIR)/include
SMAT_CUDA_LIBPATH := -L$(CUDADIR)/lib64
SMAT_CUDA_LDFLAGS := -Wl,-rpath,$(CUDADIR)/lib64
SMAT_CUDA_DIR := smat_cuda
SMAT_CUDA_CPP := $(foreach dir,$(SMAT_CUDA_DIR),$(filter-out $(dir)/cuda_autotune.cpp,$(wildcard $(dir)/*.cpp)))
SMAT_CUDA_CU  := $(foreach dir,$(SMAT_CUDA_DIR),$(filter-out $(wildcard $(dir)/*.autotune.cu),$(wildcard $(dir)/*.cu)))
SMAT_CUDA_LIB := base smat cudart cublas curand
SMAT_CUDA_DLL := $(BINDIR)/$(LIBPREFIX)smat_cuda$(LIBSUFFIX)

$(SMAT_CUDA_DLL): $(BASE_DLL) $(SMAT_DLL)
smat_cuda: $(SMAT_CUDA_DLL)

######################### libsmat_cudnn.so ###########################

SMAT_CUDNN_DEFINES := -DSMAT_CUDNN_EXPORTS
SMAT_CUDNN_INCLUDE := -I$(CUDNN_PATH) -I$(CUDADIR)/include
SMAT_CUDNN_LIBPATH := -L$(CUDNN_PATH) -L$(CUDADIR)/lib64
SMAT_CUDNN_LDFLAGS := -Wl,-rpath,$(CUDNN_PATH) -Wl,-rpath,$(CUDADIR)/lib64
SMAT_CUDNN_DIR := smat_cudnn
SMAT_CUDNN_CPP := $(foreach dir,$(SMAT_CUDNN_DIR),$(wildcard $(dir)/*.cpp))
SMAT_CUDNN_CU  := $(foreach dir,$(SMAT_CUDNN_DIR),$(wildcard $(dir)/*.cu))
SMAT_CUDNN_LIB := base smat cudart cudnn
SMAT_CUDNN_DLL := $(BINDIR)/$(LIBPREFIX)smat_cudnn$(LIBSUFFIX)

$(SMAT_CUDNN_DLL): $(BASE_DLL) $(SMAT_DLL)
smat_cudnn: $(SMAT_CUDNN_DLL)

######################### libsmat_cuda_autotune.so ###########################

SMAT_CUDA_AUTOTUNE_DEFINES := -DSMAT_CUDA_AUTOTUNE_EXPORTS
SMAT_CUDA_AUTOTUNE_INCLUDE := $(SMAT_CUDA_INCLUDE)
SMAT_CUDA_AUTOTUNE_LIBPATH := $(SMAT_CUDA_LIBPATH)
SMAT_CUDA_AUTOTUNE_LDFLAGS := $(SMAT_CUDA_LDFLAGS)
SMAT_CUDA_AUTOTUNE_DIR := $(SMAT_CUDA_DIR)
SMAT_CUDA_AUTOTUNE_CPP := $(foreach dir,$(SMAT_CUDA_AUTOTUNE_DIR),$(dir)/cuda_autotune.cpp)
SMAT_CUDA_AUTOTUNE_CU  := $(foreach dir,$(SMAT_CUDA_AUTOTUNE_DIR),$(wildcard $(dir)/*.autotune.cu))
SMAT_CUDA_AUTOTUNE_LIB := base smat smat_cuda cuda cudart
SMAT_CUDA_AUTOTUNE_DLL := $(BINDIR)/$(LIBPREFIX)smat_cuda_autotune$(LIBSUFFIX)

$(SMAT_CUDA_AUTOTUNE_DLL): $(BASE_DLL) $(SMAT_DLL) $(SMAT_CUDA_DLL)
smat_cuda_autotune: $(SMAT_CUDA_AUTOTUNE_DLL)

######################### libsmat_ext_demo.so ###########################

SMAT_EXT_DEMO_DEFINES := -DSMAT_EXT_DEMO_EXPORTS
SMAT_EXT_DEMO_INCLUDE := $(SMAT_CUDA_INCLUDE)
SMAT_EXT_DEMO_LIBPATH := $(SMAT_CUDA_LIBPATH)
SMAT_EXT_DEMO_LDFLAGS := $(SMAT_CUDA_LDFLAGS)
SMAT_EXT_DEMO_DIR := smat_ext_demo
SMAT_EXT_DEMO_CPP := $(foreach dir,$(SMAT_EXT_DEMO_DIR),$(wildcard $(dir)/*.cpp))
SMAT_EXT_DEMO_CU  := $(foreach dir,$(SMAT_EXT_DEMO_DIR),$(wildcard $(dir)/*.cu))
SMAT_EXT_DEMO_LIB := base smat smat_cuda cuda cudart
SMAT_EXT_DEMO_DLL := $(BINDIR)/$(LIBPREFIX)smat_ext_demo$(LIBSUFFIX)

$(SMAT_EXT_DEMO_DLL): $(BASE_DLL) $(SMAT_DLL) $(SMAT_CUDA_DLL)
smat_ext_demo: $(SMAT_EXT_DEMO_DLL)

####################################################################

# Tell GNU make that these are not actual file targets
.PHONY: base
.PHONY: smat
.PHONY: smat_cuda
.PHONY: smat_cudnn
.PHONY: smat_cuda_autotune
.PHONY: smat_ext_demo
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

$(eval $(call gen-module-rules,BASE))
$(eval $(call gen-module-rules,SMAT))
$(eval $(call gen-module-rules,SMAT_CUDA))
$(eval $(call gen-module-rules,SMAT_CUDNN))
$(eval $(call gen-module-rules,SMAT_EXT_DEMO))
$(eval $(call gen-module-rules,SMAT_CUDA_AUTOTUNE))

###########
# Auto-generation of header dependency rules 
# doesn't work yet, needs debugging 
#
# DEPENDS = $(patsubst %.o,%.d,$(OBJECTS))
# -include $(DEPENDS) /dev/null













####################################################################
# INSTALLATION

INSTALLLIB :=

define gen-copy-rules-so
INSTALLLIB += $(SMATPY_INST)/bin/lib$(3).so
$$($(1)_PYINST)/bin/lib$(3).so: $$($(2)DLL)
	@mkdir -p $$(@D)
	@echo copying $$< to $$@
	@cp "$$<" "$$@"
endef


#$(eval $(call gen-copy-rules-so,SMAT,BASE,base))
#$(eval $(call gen-copy-rules-so,SMAT,SMAT,smat))
#$(eval $(call gen-copy-rules-so,SMAT,SMAT_CUDA,smat_cuda))


install: $(INSTALLLIB)

