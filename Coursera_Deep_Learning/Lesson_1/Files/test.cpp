#include <iostream>
#include <cstring>
#include <assert.h>
#include <malloc.h>
using namespace std;

typedef int SLTDataType;
typedef struct SListNode
{
    SLTDataType data;
    struct SListNode* next;
}SLTNode;


// 创建节点
SLTNode* BuyListNode(SLTDataType x)
{
    SLTNode* newnode = (SLTNode*)malloc(sizeof(SLTNode));
    assert(newnode);
    newnode->data = x;
    newnode->next = NULL;

    return newnode;
}


// 找到目标值所在的地址
SLTNode* SListFind(SLTNode* phead, SLTDataType x)
{
    SLTNode* cur = phead;
    while (cur)
    {
        if (cur->data == x)
            return cur;
        
        cur = cur->next;
    }
    return NULL;
}


// 尾插节点
void SListPushBack(SLTNode** pphead, SLTDataType x)
{
    assert(pphead);
    SLTNode* newnode = BuyListNode(x);

    if (*pphead == NULL)
    {
        *pphead = newnode;
    }
    else
    {
        // 找尾节点
        SLTNode* tail = *pphead;
        while (tail->next != NULL)
        {
            tail = tail->next;
        }

        tail->next = newnode;
    }
}


//单链表在pos位置之后插入x
void SListInsertAfter(SLTNode* pos, SLTDataType x)
{
    assert(pos);
    SLTNode* newnode = BuyListNode(x);
    SLTNode* next = pos->next;
    pos->next = newnode;
    newnode->next = next;
}


// 删除目标值的节点
void SListEraseAfter(SLTNode* pos)
{
    assert(pos);
    if (pos->next == NULL)
        return;
    SLTNode* del = pos->next;
    pos->next = del->next;
    free(del);
    del = NULL;
}


// 打印链表
void SListPrint(SLTNode* phead)
{
    SLTNode* cur = phead;
    while (cur != NULL)
    {
        printf("%d->", cur->data);
        cur = cur->next;
    }
    cout << "NULL\n";
}


// 前序遍历
void pre(BTNode* p)
{
    if (p == NULL) return;
    printf("%d->", p->data);
    pre(p->left);
    pre(p->right);
}