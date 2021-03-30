/****************************************************************************************
********
********   file name:   alg_track.h
********   description: track process(match results of detect to track lists)
********   version:     V1.0
********   author:      Xu Jin
********   time:        2020-01-03 15:18
********
*****************************************************************************************/

#ifndef ALG_TRACK_H
#define ALG_TRACK_H

#include "follow_export.h"
#include "BaseList.h"

namespace MAX_algorithm {

typedef struct alg_point_
{
    int x;
    int y;
}alg_point;

typedef struct Obj_Info_Track_
{
    Obj_Info obj_info;     //detect info
    unsigned int obj_code; //obj code
    unsigned int frame_id; //current frame ID
    int          obj_state;//obj state(new/normal/...)
    int   iClassIndex;     //single or double person
    float fClassProb;

}Obj_Info_Track;

typedef struct TagObjectInfoList
{
    MAX_U32 u32ObjCode;
    MAX_U32 u32CapCode;
    MAX_S32 iInfoCount;
    MAX_S32 iDetectFailCount;
    MAX_S32 iMoveDirect;
    MAX_S32 iObjState;
    MAX_S32 iMatchScore;
    tList   tInfoList;
}tObjectInfoList;

typedef struct TagObjectInfo
{
    Obj_Box tObjBox;
    MAX_F32 fObjProb;
    MAX_S32 iObjClass;
    MAX_S32 iMatchScore;           //match score
    MAX_U32 u32FrameIndex;         //current frame index
    tObjectInfoList *pMatchList;   //point track list

}tObjectInfo;

class alg_track{
public:
    alg_track();
    ~alg_track();

    ALGErrCode Alg_Track_Init();
    ALGErrCode Alg_Track_Free();
    ALGErrCode Alg_Track_Deploy(void* srcMat, std::vector<Obj_Info>* pDetect, MAX_U32 Frame_ID, std::vector<Obj_Info_Track>* pTrack);

private:
    MAX_S32 InitMemory();
    MAX_S32 DestroyMemory();
    MAX_S32 ObjectNodeInit(MAX_S32 num,tList* pList,MAX_S32 iDataLength);
    MAX_S32 ObjectNodeQuit(tList* pList);
    MAX_S32 ObjectNodePush(void* pNode,tList* pList);
    MAX_VOID* ObjectNodePop(tList* pList);
    MAX_VOID ObjectNodeRestore(tList* pSrcList,tList* pDstList);
    MAX_VOID ObjectDetectFree(tList *DetectList,tList *InfoListFree,tList *InfoFree);

    MAX_S32 GetObjCenterDisScore(tObjectInfo *pSrcObjInfo, tObjectInfo *pObjInfo);
    MAX_S32 MatchObjInfo(tObjectInfo *pSrcObjInfo,tObjectInfo *pObjInfo);
    MAX_VOID ObjListMatch(std::vector<Obj_Info>* pDetect);
    MAX_VOID UpObjList(std::vector<Obj_Info_Track>* pTrack);
private:
    MAX_S32 m_iTaskIndex;           //track task ID
    MAX_S32 m_ImageW;
    MAX_S32 m_ImageH;
    tList   m_ObjInfoFreelist;      //free ObjInfo buf
    tList   m_ObjInfoListFreelist;  //free ObjInfoList buf
    tList   m_ObjDetectlist;        //use  ObjInfoList buf
    MAX_U32 m_u32Frame_ID;
    MAX_U32 m_u32CurObjCode;
    MAX_U32 m_u32CurCapCode;
};

}

#endif
