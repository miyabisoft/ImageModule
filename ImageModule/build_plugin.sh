#!/bin/sh
ANDROID_NDK_ROOT=/Users/macmin/ndk/android-ndk-r10e

$ANDROID_NDK_ROOT/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=Application.mk $*
mv libs/armeabi/libImageModulePlugin.so ../../../Plugins/Android

#$ANDROID_NDK_ROOT/ndk-build clean all

rm -rf libs
rm -rf obj
