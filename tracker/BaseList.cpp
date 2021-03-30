#include "BaseList.h"

//////////////////////////////////内部函数////////////////////////////////////////
static struct listnode* listnodeNew ();

/******************************************************************************
*
* listnodeAdd -Add new data to the list.
*
* RETURNS: pointer to listnode.
*/
struct listnode *
listnodeAdd (struct list2 *list, void *val)
{
    struct listnode *node;

	if ( (!list) || (!val) )
		return NULL;

	node = listnodeNew ();
	if ( !node )
		return NULL;

	node->prev = list->tail;
	node->data = val;

	if (list->head == NULL)
		list->head = node;
	else
		list->tail->next = node;
	list->tail = node;

	list->count++;

    return node;
}

/******************************************************************************
*
* listnodeDelete - Delete specific date pointer from the list. 
*
* RETURNS: N/A
*/
void
listnodeDelete (struct list2 *list, void *val)
{
    struct listnode *node;

	if ( (!list) || (!val) )
		return;

	for (node = list->head; node; node = node->next)
	{
		if (node->data == val)
		{
			if (node->prev)
				node->prev->next = node->next;
			else
				list->head = node->next;

			if (node->next)
				node->next->prev = node->prev;
			else
				list->tail = node->prev;

			list->count--;
			listnodeFree (node);

			return;
		}
	}
}

/******************************************************************************
*
* listnodeHead - Return first node's data if it is there. 
*
* RETURNS: N/A
*/
void *
listnodeHead (struct list2 *list)
{
    struct listnode *node;
	node = list->head;

	if (node)
		return node->data;
	return NULL;
}

/******************************************************************************
*
* listnodeHead - Return first node's data if it is there. 
*
* RETURNS: N/A
*/
void *
listnodeTail (struct list2 *list)
{
    struct listnode *node;

	node = list->tail;

	if (node)
		return node->data;
	return NULL;
}

/******************************************************************************
*
* listNew -Allocate new list.
*
* RETURNS: pointer to list.
*/
struct list2 *
listNew ()
{
    struct list2 *newList = NULL;
    newList = (struct list2 *)calloc(1, sizeof (struct list2));
	if(newList!=NULL)
	{
		newList->head = NULL ;
		newList->tail = NULL ;
		newList->count = 0 ;

	}

	return newList;
}

/******************************************************************************
*
* listFree -Free list.
*
* RETURNS: N/A.
*/
void
listFree (struct list2 *l)
{
	if(l!=NULL)
	{
		free ( l);
	}
}

/******************************************************************************
*
* listDelete - Delete all listnode then free list itself. 
*
* RETURNS: N/A
*/
void
listDelete (struct list2 *list)
{
	struct listnode *node;
	struct listnode *next;

	for (node = list->head; node; node = next)
	{
		next = node->next;

		listnodeFree (node);
	}
    listFree (list);
}

/******************************************************************************
*
* listnodeHead - Delete all listnode from the list.
*
* RETURNS: N/A
*/
void
listDeleteAllNode (struct list2 *list)
{
	struct listnode *node;
	struct listnode *next;

	for (node = list->head; node; node = next)
	{
		next = node->next;
		listnodeFree (node);
	}
	list->head = list->tail = NULL;
	list->count = 0;
}

/******************************************************************************
*
* listDeleteNode - Delete the node from list.  
*
* RETURNS: N/A.
*/
void
listDeleteNode (struct list2 *list, struct listnode *node)
{
	if (node->prev)
		node->prev->next = node->next;
	else
		list->head = node->next;
	if (node->next)
		node->next->prev = node->prev;
	else
		list->tail = node->prev;
	list->count--;
	listnodeFree (node);
}


void* 
listPop(struct list2 *list)
{
	void *data = NULL;

	struct listnode *node=NULL;

	node = list->head;

	if ( node != NULL )
	{
		data = node->data ; 

		if (node->prev)
			node->prev->next = node->next;
		else
			list->head = node->next;
		if (node->next)
			node->next->prev = node->prev;
		else
			list->tail = node->prev;

		list->count--;
		listnodeFree (node);
	}

	return data;
}


/******************************************************************************
*
* listnodeFree -free listnode.
*
* RETURNS: N/A.
*/
void
listnodeFree (struct listnode *node)
{
	if(node!=NULL)
		free ( node);
}


//////////////////////////////////内部函数////////////////////////////////////////



/******************************************************************************
*
* listnodeNew -Allocate new listnode.
*
* RETURNS: pointer to listnode.
*/
static struct listnode *
listnodeNew ()
{
	struct listnode *node;

	node = (struct listnode *)calloc( 1,sizeof (struct listnode));
	if(node != NULL)
	{
		node->prev = NULL ;
		node->next = NULL ;
		node->data = NULL ;
	}

	return node;
}

