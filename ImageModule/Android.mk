include $(CLEAR_VARS)

#opencv
OPENCVROOT:= /Users/macmin/Documents/opencv2/OpenCV-android-sdk
OPENCV_CAMERA_MODULES:=off
OPENCV_INSTALL_MODULES:=on
#OPENCV_LIB_TYPE:=SHARED
OPENCV_LIB_TYPE:=STATIC
include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

LOCAL_ARM_MODE := arm
LOCAL_PATH := $(NDK_PROJECT_PATH)
LOCAL_MODULE := ImageModulePlugin
LOCAL_CFLAGS := -Werror
LOCAL_SRC_FILES := Imagesmodule.cpp LanguagePlugin.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_LDLIBS     += -llog -ldl -lGLESv1_CM -lGLESv2

include $(BUILD_SHARED_LIBRARY)
