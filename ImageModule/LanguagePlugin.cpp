//
//  LanguagePlugin.cpp
//  ImageModule
//
//  Created by setomasakazu on 2015/06/27.
//  Copyright (c) 2015年 setomasakazu. All rights reserved.
//

#include "LanguagePlugin.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

char* MakeStringCopy (const char* string) {
    if (string == NULL)
        return NULL;
    
    char* res = (char*)malloc(strlen(string) + 1);
    strcpy(res, string);
#ifdef __ANDROID__
    __android_log_write(ANDROID_LOG_DEBUG, "MakeStringCopy", res);
#else
    printf("%s %s", "MakeStringCopy",res);
#endif
    return res;
}

char* CurrentLanguage() {
    return MakeStringCopy("This iOS1 & Android");
}

