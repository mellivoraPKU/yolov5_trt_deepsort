#ifndef __MAXVISION_H__
#define __MAXVISION_H__

typedef char 				MAX_SC;
typedef unsigned char		MAX_UC;

typedef char 				MAX_S8;
typedef unsigned char		MAX_U8;

typedef short				MAX_S16;
typedef unsigned short		MAX_U16;

typedef int 				MAX_S32;
typedef unsigned int		MAX_U32;

typedef long long			MAX_S64;
typedef unsigned long long 	MAX_U64;

typedef float 				MAX_F32;
typedef double 				MAX_F64;

typedef unsigned int 		MAX_BOOL;

typedef void				MAX_VOID;

typedef MAX_VOID *			MAX_HANDLE;

#define MAX_TRUE  1
#define MAX_FALSE 0
#define MAX_OK    0
#define MAX_FAIL -1

#ifndef __ERROR_CODE__
#define __ERROR_CODE__
typedef int 		Error_Code;
#endif //__ERROR_CODE__

#endif // __MAXVISION_H__

