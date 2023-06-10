################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm

# All of the sources participating in the build are defined here
-include sources.mk
-include src/subdir.mk
-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := 2dConvolutions
BUILD_ARTIFACT_EXTENSION :=
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: main-build

# Main-build Target
main-build: 2dConvolutions

# Tool invocations
2dConvolutions: $(OBJS) $(USER_OBJS) makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/usr/local/cuda-12.1/bin/nvcc --cudart=static -L/usr/lib -L/usr/lib64 -ccbin g++ -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o "2dConvolutions" $(OBJS) $(USER_OBJS) $(LIBS) -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_ml -lopencv_videoio -lopencv_video -lopencv_features2d
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) 2dConvolutions
	-@echo ' '

.PHONY: all clean dependents main-build

-include ../makefile.targets
