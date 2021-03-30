#ifndef FOLLOW_EXPORT_H
#define FOLLOW_EXPORT_H

#include "maxvision_type.h"
#include "algorithm_sdk_error_code.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if(defined WIN32||defined_WIN32|| defined WINCE)
#define FOLLOW_DLL_EXPORT __declspec(dllexport)
#else
#define FOLLOW_DLL_EXPORT
#endif

namespace MAX_algorithm {

typedef struct Obj_Box_
{
    int x;
    int y;
    int width;
    int height;
}Obj_Box;

typedef struct Obj_Info_
{
    Obj_Box obj_box;
    float   obj_prob;
    int     obj_class;
}Obj_Info;

#define MAX_DETECTOBJ_NUM 100
typedef struct ObjDetectResult_
{
    Obj_Info obj_info[MAX_DETECTOBJ_NUM];
    int  obj_num;

    int   iClassIndex;     //single or double person
    float fClassProb;
}ObjDetectResult;

typedef struct Image_Uniformization_
{
    MAX_S32 width;
    MAX_S32 height;
    MAX_S32 channel;
    MAX_F32 *image_data;
    MAX_S32 src_image_height;// 原始图像长
    MAX_S32 src_image_width;// 原始图像宽
}Image_Uniformization;

typedef enum
{
    Param_Detect_ROI,               //param type: Obj_Box
    Param_Detect_Threld,            //param type: float
    Param_Detect_Other
}eFollow_ParamType;


/***************************************************************
funcName  : FOLLOW_GetVersion
funcs	  : Get ALG version
Param In  : algHandle--detect handle;
Prame Out : version--alg version
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode  FOLLOW_GetVersion(char** version, MAX_HANDLE algHandle);

////////////////////////////////alg detect start ///////////////////////////////

/***************************************************************
funcName  : FOLLOW_AlgCreate
funcs	  : create ALG handle
Param In  : model -- model path; pLicense -- license;
Param Out : pHandle -- alg handle;
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_AlgCreate(MAX_HANDLE *pHandle, MAX_S8* model, MAX_S8* modelArbitration, MAX_S8* pLicense, Algorithm_Lib_License_mode* eLicenseMode);

/***************************************************************
funcName  : FOLLOW_AlgFree
funcs	  : free ALG handle
Param In  : pHandle -- alg handle;
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_AlgFree(MAX_HANDLE *pHandle);

/***************************************************************
funcName  : FOLLOW_SetParam
funcs	  : set ALG param
Param In  : Handle -- alg handle; eType -- param type; pParam -- param;
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_SetParam(MAX_HANDLE Handle, eFollow_ParamType eType, MAX_VOID* pParam);

/***************************************************************
funcName  : FOLLOW_GetParam
funcs	  : get ALG param
Param In  : Handle -- alg handle; eType -- param type; 
Param Out : pParam -- param;
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_GetParam(MAX_HANDLE Handle, eFollow_ParamType eType, MAX_VOID* pParam);

/***************************************************************
funcName  : FOLLOW_fillData
funcs	  : change bgr image to ALG needed image;
Param In  : srcImage -- bgr image;  
Param Out : outImage -- Alg needed image;
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_fillData(MAX_HANDLE Handle, MAX_U8* srcImage, int iSrcW, int iSrcH, Image_Uniformization* outImage);

/***************************************************************
funcName  : FOLLOW_freeData
funcs	  : free bgr image which ALG needed image;
Param In  : Handle -- alg handle; srcImage -- bgr image;
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_freeData(MAX_HANDLE Handle, Image_Uniformization* outImage);




/***************************************************************
funcName  : FOLLOW_forword
funcs	  : get alg detect result of one frame;
Param In  : Handle -- alg handle; inputImage -- uniform image; srcImage -- bgr image; iSrcW -- src width; src height;
Param Out : result -- detect results (ObjDetectResult*);
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_forword(MAX_HANDLE Handle, Image_Uniformization* inputImage, MAX_U8* srcImage, int iSrcW, int iSrcH, const float confidence_thresh ,const float nms_thresh, ObjDetectResult* result);

////////////////////////////////alg detect end //////////////////////////////////

////////////////////////////////alg arbitration start////////////////////////////
typedef enum
{
    IMAGE_QUALITY_NORMAL     = 0, //normal
    IMAGE_QUALITY_BLUR       = 1, //image blur
    IMAGE_QUALITY_OCCLUSIVE  = 2, //image occlusive
    IMAGE_QUALITY_SHIFT      = 3,  //image shift
    IMAGE_QUALITY_OFFLINE    = 4

}eImageQualityState;

typedef enum
{
    Arbitration_Follow,       //detect follow or not
    Arbitration_Burst,        //detect burst or not
    Arbitration_Hangover,     //detect hangover or not
    Arbitration_ImageQuality, //detect image quality
    Arbitration_ImageOffline
}eArbitrationType;

typedef struct ArbitrationResult_
{
    int   iFollowState;            //0 -- no person; 1 -- normal; 2 -- follow;
    float fFollowConfidence;       //confidence of follow;
    int   iBurstState;             //0 -- normal; 1 -- burst;
    float fBurstConfidence;        //confidence of burst;
    int   iHangeState;             //0 -- normal; 1 -- hangover;
    float fHangeConfidence;        //confidence of hangover;
    int   iImageQualityState;      //0 -- normal; not 0 -- unnormal;
    float fImageQualityConfidence; //confidence of image quality;
}ArbitrationResult;

typedef enum
{
    Arbitration_Trailing_Info,
    Arbitration_Intrude_Info,
    Arbitration_Camera_Check_Info,
    Arbitration_other
}eFollow_ParamTypeArbitration;


typedef struct Roi_Param_
{
    MAX_S32 door1X;  //left
    MAX_S32 door2X;  //right
    MAX_S32 fence1Y; //top
    MAX_S32 fence2Y; //bottom

}Roi_Param;

typedef struct Camera_Roi_Param_
{
    MAX_S32 x;
    MAX_S32 y;
    MAX_S32 width;
    MAX_S32 height;

}Camera_Roi_Param;
typedef struct Trailing_Analysis_Param_
{
    MAX_F32 class_thresh;
    MAX_F32 nms_thresh;
    MAX_U32 test_frame;
    MAX_U32 strategy1_count;
    MAX_F32 strategy1_thresh;
    MAX_U32 strategy2_count;
    MAX_F32 strategy2_thresh;
    MAX_U32 strategy3_count;
    MAX_F32 strategy3_thresh;
    /* 逃逸参数 */
    MAX_U32 escape_count;
    MAX_F32 escape_thresh;

    Roi_Param roi_param;

}Trailing_Analysis_Param;

typedef struct Intrude_Analysis_Param_
{
    MAX_U32 test_frame;
    MAX_U32 intrude_count;
    MAX_F32 intrude_thresh;
    Roi_Param roi_param;

}Intrude_Analysis_Param;

typedef struct Camrea_Check_Param_
{
    MAX_F32 blur_thresh;
    MAX_F32 occlusive_thresh;
    MAX_F32 shift_thresh;
    Camera_Roi_Param roi_param;

}Camrea_Check_Param;


/***************************************************************
funcName  : FOLLOW_ArbitrationCreate
funcs	  : create Arbitration handle
Param In  :
Param Out : pHandle -- alg handle;
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_ArbitrationCreate(MAX_HANDLE *pHandle);


/***************************************************************
funcName  : FOLLOW_ArbitrationFree
funcs	  : free Arbitration handle
Param In  : pHandle -- Arbitration handle;
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_ArbitrationFree(MAX_HANDLE *pHandle);

/***************************************************************
funcName  : FOLLOW_SetParamArbitration
funcs	  : set Arbitration param
Param In  : Handle -- Arbitration handle; eType -- param type; pParam -- (Type:void*);
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_ArbitrationSetParam(MAX_HANDLE Handle, eFollow_ParamTypeArbitration eType, void* pParam);

/***************************************************************
funcName  : FOLLOW_GetParamArbitration
funcs	  : get Arbitration param
Param In  : Handle -- Arbitration handle; eType -- param type; 
Param Out : pParam -- ArbitrationResult_Param(union);
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_ArbitrationGetParam(MAX_HANDLE Handle, eFollow_ParamTypeArbitration eType, void* pParam);

/***************************************************************
funcName  : FOLLOW_ProcessArbitration
funcs	  : get result of Arbitration;
Param In  : Handle -- Arbitration handle; srcImage -- bgr image; iSrcW -- src width; iSrcH -- src height; detectResult -- result of detect Alg; eType -- Arbitration Type;
Param Out : outResult -- result of Arbitration;
return    : 0 -- success  other -- fail
**************************************************************/
FOLLOW_DLL_EXPORT ALGErrCode FOLLOW_ArbitrationProcess(MAX_HANDLE Handle, MAX_U8* srcImage, MAX_S32 iSrcW, MAX_S32 iSrcH, MAX_U32 Frame_ID, ObjDetectResult* detectResult, eArbitrationType eType, ArbitrationResult* outResult);

////////////////////////////////alg arbitration end//////////////////////////////

}
#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif


