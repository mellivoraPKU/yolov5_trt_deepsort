#include <string.h>
#include <stdio.h>

#include "alg_track.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

namespace MAX_algorithm {

#define MEMO_FREE(p)\
{\
    if( p )\
{\
    free(p);\
    p=NULL;\
}\
}
#define MEMO_MALLOC(p, type, size)\
    {\
    MEMO_FREE(p);\
    p = (type *) malloc( (size)*sizeof(type));\
}


alg_track::alg_track()
{
    m_iTaskIndex = 0;
    m_ImageW = 1920;
    m_ImageH = 1080;
    memset(&m_ObjInfoFreelist, 0, sizeof(tList));
    memset(&m_ObjInfoListFreelist, 0, sizeof(tList));
    memset(&m_ObjDetectlist, 0, sizeof(tList));
    m_u32Frame_ID = 0;
    m_u32CurObjCode = 0;
    m_u32CurCapCode = 0;
}

alg_track::~alg_track()
{

}



/***************************************************************
funcName  : Alg_Track_Init
funcs	  : Alg init
Param In  : pHandle--Alg Handle
Param Out : NULL
return    : 0 -- success  other -- fail
**************************************************************/
ALGErrCode alg_track::Alg_Track_Init()
{
    InitMemory();

    return ALGORITHM_OPERATION_SUCCESS;
}

/***************************************************************
funcName  : Alg_Track_Free
funcs	  : Alg free
Param In  : pHandle--Alg Handle
Param Out : NULL
return    : 0 -- success  other -- fail
**************************************************************/
ALGErrCode alg_track::Alg_Track_Free()
{
    DestroyMemory();

    return ALGORITHM_OPERATION_SUCCESS;
}


/***************************************************************
funcName  : Alg_Track_Deploy
funcs	  : Alg deloy
Param In  : Handle--Alg Handle; srcMat--src image; Frame_ID--current FrameID;
Param Out : pTrack--track results
return    : 0 -- success  other -- fail
**************************************************************/
ALGErrCode alg_track::Alg_Track_Deploy(void* srcMat, std::vector<Obj_Info>* pDetect, MAX_U32 Frame_ID, std::vector<Obj_Info_Track>* pTrack)
{
    if(NULL == srcMat || NULL == pDetect || NULL == pTrack)
    {
        return ALGORITHM_PARAM_ERR;
    }
    Mat* src = (Mat*)srcMat;

    m_ImageW = src->cols;
    m_ImageH = src->rows;
    m_u32Frame_ID = Frame_ID;

    ObjListMatch(pDetect);

    UpObjList(pTrack);

    return ALGORITHM_OPERATION_SUCCESS;
}


//////////////////////////private//////////////////////////////

/***************************************************************
funcName  : ObjectNodeInit
funcs	  : init list
Param In  : num--node num; pList--List; iDataLength--node data size;
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
MAX_S32 alg_track::ObjectNodeInit(MAX_S32 num,tList* pList,MAX_S32 iDataLength)
{
    MAX_S32 i = 0;
    void *pNode = NULL;
    for (i = 0; i < num; i++)
    {
        MEMO_MALLOC(pNode,MAX_S8,iDataLength);
        if (pNode != NULL)
        {
            listnodeAdd(pList, pNode);
            pNode = NULL;
        }
        else
        {
            return MAX_FAIL;
        }
    }
    return MAX_OK;
}

/***************************************************************
funcName  : ObjectNodeQuit
funcs	  : destroy list
Param In  : pList--List;
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
MAX_S32 alg_track::ObjectNodeQuit(tList* pList)
{
    MAX_VOID *pNode = NULL;

    while(1)
    {
        pNode = listPop(pList);

        if (pNode != NULL)
        {
            MEMO_FREE(pNode);
            pNode = NULL;
        }
        else
        {
            break;
        }
    }
    return MAX_OK;
}

//push to list
MAX_S32 alg_track::ObjectNodePush(void* pNode,tList* pList)
{
    listnodeAdd(pList, pNode);
    return MAX_OK;
}

//pop from list
MAX_VOID* alg_track::ObjectNodePop(tList* pList)
{
    MAX_VOID *pNode = NULL;

    pNode = listPop(pList);

    return pNode;
}

//restore node from SrcList to DstList
MAX_VOID alg_track::ObjectNodeRestore(tList* pSrcList,tList* pDstList)
{
    MAX_VOID *pNode = NULL;
    while (1)
    {
        pNode = ObjectNodePop(pSrcList);

        if (pNode)
        {
            ObjectNodePush(pNode,pDstList);
        }
        else
        {
            break;
        }
    }
}

MAX_VOID alg_track::ObjectDetectFree(tList *DetectList,tList *InfoListFree,tList *InfoFree)
{
    tObjectInfoList *pNode = NULL;
    struct listnode *node = NULL;
    void *data = NULL;
    //restore all info
    LIST_LOOP(DetectList,data,node)
    {
        pNode = (tObjectInfoList*)data;
        ObjectNodeRestore(&pNode->tInfoList,InfoFree);
    }
    //restore all infoList
    ObjectNodeRestore(DetectList,InfoListFree);
}

/***************************************************************
funcName  : InitMemory
funcs	  : malloc buf
Param In  :
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
MAX_S32 alg_track::InitMemory()
{
    ObjectNodeInit(2000, &m_ObjInfoFreelist,     sizeof(tObjectInfo));
    ObjectNodeInit(100,  &m_ObjInfoListFreelist, sizeof(tObjectInfoList));

    return MAX_OK;
}

/***************************************************************
funcName  : DestroyMemory
funcs	  : free buf
Param In  :
Param Out :
return    : 0 -- success  other -- fail
**************************************************************/
MAX_S32 alg_track::DestroyMemory()
{
    ObjectDetectFree(&m_ObjDetectlist,&m_ObjInfoListFreelist,&m_ObjInfoFreelist);
    ObjectNodeQuit(&m_ObjInfoFreelist);
    ObjectNodeQuit(&m_ObjInfoListFreelist);

    return MAX_OK;
}


//compute IOU
static float OverLap(float x1,float x2,float y1,float y2)
{
    float left = (x1>y1?x1:y1);
    float right = (x2<y2?x2:y2);

    return right-left;
}

static float box_intersection(Obj_Box *a,Obj_Box *b)
{
    float w = OverLap(a->x,a->x+a->width,b->x,b->x+b->width);
    float h = OverLap(a->y,a->y+a->height,b->y,b->y+b->height);

    if (w < 0 || h < 0)
    {
        return 0.0;
    }
    else
    {
        return (w*h);
    }
}

static float GetRectIOU(Obj_Box *a,Obj_Box *b)
{
    float InValue = box_intersection(a,b);
    float UnValue = (a->width*a->height+b->width*b->height-InValue);

    return InValue/UnValue;
}

/***************************************************************
funcName  : MatchObjInfo
funcs	  : match objInfo
Param In  : pSrcObjInfo--pSrcObj; pObjInfo--pDstObj;
Param Out :
return    : match score
**************************************************************/
MAX_S32 alg_track::MatchObjInfo(tObjectInfo *pSrcObjInfo,tObjectInfo *pObjInfo)
{
    MAX_S32 iMatchScore = 0;

    iMatchScore = 100*GetRectIOU(&pSrcObjInfo->tObjBox,&pObjInfo->tObjBox);

    return iMatchScore;
}

MAX_S32 alg_track::GetObjCenterDisScore(tObjectInfo *pSrcObjInfo, tObjectInfo *pObjInfo)
{
    int SrcX,SrcY,DstX,DstY;
    double CenterDIS = 0;
    SrcX = pSrcObjInfo->tObjBox.x + pSrcObjInfo->tObjBox.width/2;
    SrcY = pSrcObjInfo->tObjBox.y + pSrcObjInfo->tObjBox.height/2;

    DstX = pObjInfo->tObjBox.x + pObjInfo->tObjBox.width/2;
    DstY = pObjInfo->tObjBox.y + pObjInfo->tObjBox.height/2;

    CenterDIS = sqrt((SrcX-DstX)*(SrcX-DstX)+(SrcY-DstY)*(SrcY-DstY));
    CenterDIS = 10000/CenterDIS;
    return CenterDIS;
}

/***************************************************************
funcName  : ObjListMatch(follow only has obj one or two)
funcs	  : match all objInfo to Lists
Param In  : pDetect--detect results;
Param Out :
return    :
**************************************************************/
MAX_VOID alg_track::ObjListMatch(std::vector<Obj_Info>* pDetect)
{
    struct listnode *node = NULL;
    void *data = NULL;
    tObjectInfoList *pObjInfoList = NULL;
    tObjectInfo *pObjNode = NULL;
    tObjectInfo *pTailInfo = NULL;
    MAX_S32 iMatchIndex = -1,iMatchScore = 0, iMatchScoreMax = 0;
    MAX_U32 u32FrameID = 0;
    MAX_S32 iObjectFlag[MAX_DETECTOBJ_NUM];
    MAX_S32 iObjectNum = 0;

    iObjectNum = (pDetect->size()<MAX_DETECTOBJ_NUM?pDetect->size():MAX_DETECTOBJ_NUM);
    iObjectNum = MIN(30, iObjectNum); //only use most two detect objects
    memset(iObjectFlag, 0 ,MAX_DETECTOBJ_NUM*sizeof(MAX_S32));

    u32FrameID = m_u32Frame_ID;

    //LOOP DetectList
    LIST_LOOP(&m_ObjDetectlist, data, node)
    {
        pObjInfoList = (tObjectInfoList*)data;

        m_u32CurObjCode = pObjInfoList->u32ObjCode;
        pObjNode = (tObjectInfo*)ObjectNodePop(&m_ObjInfoFreelist);
        if(NULL == pObjNode)
        {
            continue;
        }
        pTailInfo = (tObjectInfo*)listnodeTail(&pObjInfoList->tInfoList);
        iMatchIndex = -1;
        iMatchScoreMax = 0;
        //to match detect objs
        for(int i = 0; i < iObjectNum; ++i)
        {
            if(iObjectFlag[i] == 1)
            {//has been match ok
                continue;
            }
            //get match score
            pObjNode->tObjBox = (*pDetect)[i].obj_box;

            iMatchScore = MatchObjInfo(pTailInfo, pObjNode);
            if(iMatchScore > iMatchScoreMax)
            {
                iMatchScoreMax = iMatchScore;
                iMatchIndex = i;
            }
        }

//        if(iObjectNum > 0 && iMatchScoreMax <=0)
//        {//find min dis of center(iMatchScore)
//            for(int i = 0; i < iObjectNum; ++i)
//            {
//                if(iObjectFlag[i] == 1)
//                {
//                    continue;
//                }
//                pObjNode->tObjBox = (*pDetect)[i].obj_box;

//                iMatchScore = GetObjCenterDisScore(pTailInfo, pObjNode);
//                if(iMatchScore > iMatchScoreMax)
//                {
//                    iMatchScoreMax = iMatchScore;
//                    iMatchIndex = i;
//                }
//            }

//        }

        if(iMatchScoreMax >30)
        {
            iObjectFlag[iMatchIndex] = 1;
            pObjInfoList->iDetectFailCount = 0;
            pObjInfoList->iMatchScore = iMatchScoreMax;
            pObjInfoList->iInfoCount++;

            pObjNode->tObjBox     = (*pDetect)[iMatchIndex].obj_box;
            pObjNode->fObjProb    = (*pDetect)[iMatchIndex].obj_prob;
            pObjNode->iObjClass   = (*pDetect)[iMatchIndex].obj_class;
            pObjNode->iMatchScore = iMatchScoreMax;
            pObjNode->u32FrameIndex = u32FrameID;
            pObjNode->pMatchList = pObjInfoList;

            ObjectNodePush(pObjNode, &pObjInfoList->tInfoList);
        }
        else
        {
            ObjectNodePush(pObjNode, &m_ObjInfoFreelist);
        }
    }

    //add new obj to m_objDetectlist
    for(int i = 0; i < iObjectNum; ++i)
    {
        if(iObjectFlag[i] == 1)
        {//has been match ok
            continue;
        }
        pObjNode = (tObjectInfo*)ObjectNodePop(&m_ObjInfoFreelist);
        if(NULL == pObjNode)
        {
            continue;
        }
        pObjNode->tObjBox     = (*pDetect)[i].obj_box;
        pObjNode->fObjProb    = (*pDetect)[i].obj_prob;
        pObjNode->iObjClass   = (*pDetect)[i].obj_class;
        pObjNode->iMatchScore = 0;
        pObjNode->u32FrameIndex = u32FrameID;
        pObjNode->pMatchList = NULL;
        pObjInfoList = (tObjectInfoList*)ObjectNodePop(&m_ObjInfoListFreelist);

        if(pObjInfoList)
        {
            memset(pObjInfoList, 0, sizeof(tObjectInfoList));
            ObjectNodePush(pObjNode, &pObjInfoList->tInfoList);

           // m_u32CurObjCode = (m_u32CurObjCode == 0?1:0);

             m_u32CurObjCode++;
            pObjInfoList->u32ObjCode = m_u32CurObjCode;

            pObjInfoList->iInfoCount = 1;
            pObjInfoList->iDetectFailCount = 0;
            pObjInfoList->iMatchScore = 0;
            ObjectNodePush(pObjInfoList, &m_ObjDetectlist);
        }
        else
        {

            ObjectNodePush(pObjNode, &m_ObjInfoFreelist);
        }
    }
}


/***************************************************************
funcName  : UpObjList
funcs	  : updata list
Param In  :
Param Out : pTrack--track results
return    :
**************************************************************/
MAX_VOID alg_track::UpObjList(std::vector<Obj_Info_Track>* pTrack)
{
    MAX_S32 i = 0;
    tObjectInfoList* pObjInfoList = NULL;
    struct listnode* node         = NULL;
    struct listnode* nextNode     = NULL;
    struct listnode* preNode      = NULL;
    tObjectInfo* tempObj = NULL;
    MAX_S32 iDetectThreld;

    iDetectThreld =5;

    pTrack->clear();

    node = m_ObjDetectlist.head;

    while (node&&node->data)
    {
        nextNode = node->next;

        pObjInfoList = (tObjectInfoList*)node->data;

        tempObj = (tObjectInfo*)listnodeTail(&pObjInfoList->tInfoList);

        if (pObjInfoList->iDetectFailCount>iDetectThreld)
        {
            preNode = node->prev;
            ObjectNodeRestore(&pObjInfoList->tInfoList,&m_ObjInfoFreelist);
            memset(pObjInfoList,0,sizeof(tObjectInfoList));
            ObjectNodePush(pObjInfoList,&m_ObjInfoListFreelist);

            if (preNode)
            {
                if (nextNode)
                {
                    nextNode->prev = preNode;
                }
                preNode->next = nextNode;
            }
            else
            {
                if (nextNode)
                {
                    nextNode->prev = NULL;
                }
            }
            if (preNode==NULL)
            {
                m_ObjDetectlist.head = nextNode;
            }
            if (nextNode==NULL)
            {
                m_ObjDetectlist.tail = preNode;
            }
            m_ObjDetectlist.count--;

            listnodeFree(node);

        }
        else
        {
            pObjInfoList->iDetectFailCount++;
         //  if(pObjInfoList->iObjState)
            {
                Obj_Info_Track tTrackObject;
                tTrackObject.obj_info.obj_box   = tempObj->tObjBox;
                tTrackObject.obj_info.obj_prob  = tempObj->fObjProb;
                tTrackObject.obj_info.obj_class = tempObj->iObjClass;
                tTrackObject.obj_code = pObjInfoList->u32ObjCode;
                //tTrackObject.obj_code = pObjInfoList->u32CapCode;
                tTrackObject.frame_id = tempObj->u32FrameIndex;
                tTrackObject.obj_state = pObjInfoList->iObjState;

                pTrack->push_back(tTrackObject);
            }
        }

        node = nextNode;
    }
}




}


