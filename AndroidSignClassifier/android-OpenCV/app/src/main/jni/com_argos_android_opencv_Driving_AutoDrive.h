/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_argos_android_opencv_Driving_AutoDrive */

#ifndef _Included_com_argos_android_opencv_Driving_AutoDrive
#define _Included_com_argos_android_opencv_Driving_AutoDrive
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_argos_android_opencv_Driving_AutoDrive
 * Method:    drive
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_argos_android_opencv_Driving_AutoDrive_drive
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_argos_android_opencv_Driving_AutoDrive
 * Method:    detectVehicle
 * Signature: (Ljava/lang/String;J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_argos_android_opencv_Driving_AutoDrive_detectVehicle
  (JNIEnv *, jclass, jstring, jlong);

/*#include "tensorflow/cc/client/client_session.h"
 * Class:     com_argos_android_opencv_Driving_AutoDrive
 * Method:    detectSpeedLimit
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_argos_android_opencv_Driving_AutoDrive_detectSpeedLimit
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif
