#ifndef ALGORITHM_SDK_ERROR_CODE_H
#define ALGORITHM_SDK_ERROR_CODE_H
namespace MAX_algorithm {
    typedef enum
    {
        ALGORITHM_LIB_LICENSE_UNLAWFULNESS    = 0,    /* 无效 */
        ALGORITHM_LIB_LICENSE_PROBATION       = 1,    /* 试用 */
        ALGORITHM_LIB_LICENSE_TARGET_DATE     = 2,    /* 一段时间内有效 */
        ALGORITHM_LIB_LICENSE_LIFETIME        = 3,    /* 终身许可 */
        ALGORITHM_LIB_LICENSE_END
    }Algorithm_Lib_License_mode;


    /*error code*/
    typedef enum
    {
        ALGORITHM_OPERATION_SUCCESS                      =  0,
        /******************init*****************************/
        ALGORITHM_INIT_ERR                     = (  01 ),
        ALGORITHM_INIT_REPETITION_ERR          = (  02 ),  /*重复初始化模型*/
        ALGORITHM_INIT_MODEL_ERR               = (  03 ),  /*模型加载出错*/
        ALGORITHM_WRONG_IMAGE_ERR              = (  04 ),/*not support this net*/
        ALGORITHM_INIT_PROTOTXT_ERR            = (  05 ),/*未知的postprocess模式设置 */
        ALGORITHM_WRONG_DETECTOR_ERR           = (  06 ),/*postprocess代码没有实现*/
        ALGORITHM_ERROR_FUNC_MEM_ENOUGH        = (  07 ),
        ALGORITHM_POSTPROCESS_FAILE            = (  8  ),
        /***************** 图片数据准备 **********************/
        ALGORITHM_IMAGE_UNIFORMAZATION_ERR     = (  9  ),
        ALGORITHM_ALGORITHM_DELETE_ERR         = (  10 ),
        /***********************************************/


        ALGORITHM_INIT_FIRMWARE_HEAD_ERR				= (  11 ),		/* 固件头错误 */
        ALGORITHM_FIRMWARE_VERSION_MAIN_ERR			    = (  12 ),		/* 加密固件软件主版本不一致 */
        ALGORITHM_FIRMWARE_VERSION_MID_ERR			    = (  13 ),		/* 加密固件软件中版本不一致 */
        ALGORITHM_FIRMWARE_VERSION_MAINMAIN_ERR		    = (  14 ),		/* 加密固件软件低版本不一致 */
        ALGORITHM_FIRMWARE_FILE_LENGTH_ERR			    = (  15 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_ENC_CHECK_ERR           = (  16 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_SRC_CHECK_ERR           = (  17 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_TYPE_ERR				= (  18 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_SAVE_ERR				= (  19 ),		/*  */
        ALGORITHM_ERROR_FUNC_PARAM_LAW                  = (  20 ),		/*  */
        ALGORITHM_NEW_ALGORITHM_ENCRYPTION_FAIL		    = (  21 ),		/*  */
        ALGORITHM_UNAUTHORZED				            = (  22 ),		/* 未授权 */
        ALGORITHM_KEY_OVER_PROABTION					= (  23 ),		/* 试用时间过期 */
        ALGORITHM_KEY_OVERDUE                           = (  24 ),		/* 秘钥过期 */
        ALGORITHM_LICENSE_FRMAT_ILL					    = (  30 ),		/* 秘钥格式非法 */
        ALGORITHM_LICENSE_CPU_CHECK					    = (  31 ),		/* 芯片ID验证失败 */
        ALGORITHM_FIRMWARE_AL_VERSION_HEAD	            = (  32 ),		/*  无法获取算法模型的版本号*/
        ALGORITHM_INIT_FIRMWARE_NOT_EXIT				= (  33 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_COUNT_ERR				= (  34 ),		/*  */
        ALGORITHM_KEY_VERIFY_FAIL						= (  35 ),		/*  */
        ALGORITHM_ERROR_FUNC_FILE_READ                  = (  36 ),		/* 无法读取固件 */
        ALGORITHM_OPEN_CHIP_ID_ERR                      = (  37 ),		/* 获取芯片ID失败 */
        ALGORITHM_PARAM_ERR                             = (  38 ),
        ALGORITHM_INIT_LOAD_CUSTOM_OP_ERR               = (  39 ),      /*TX2平台加载custom OP错误*/
        ALGORITHM_ERR_OTHER                             =  999
    } ALGErrCode;

}
#endif // ALGORITHM_SDK_ERROR_CODE_H
