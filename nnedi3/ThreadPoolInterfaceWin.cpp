#include "ThreadPoolInterfaceWin.h"

#if defined(_WIN32) || defined(_WIN64)

#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}
#define mydelete(ptr) if (ptr!=NULL) { delete ptr; ptr=NULL;}

ThreadPoolInterfaceWin::ThreadPoolInterfaceWin(void): ThreadPoolInterfaceBase(),
    CSectionOk(FALSE), ghMutexResources(NULL)
{
    CSectionOk = InitializeCriticalSectionAndSpinCount(&CriticalSection, 0x00000400);
    if (CSectionOk == TRUE)
    {
        ghMutexResources = CreateMutex(NULL, FALSE, NULL);
        if (ghMutexResources == NULL)
        {
            CSectionOk = FALSE;
            DeleteCriticalSection(&CriticalSection);
        }
        else Status_Ok = true;
    }

    EndExclusive = NULL;

    for (uint8_t i = 0; i < MAX_THREAD_POOL; i++)
    {
        JobsEnded[i] = NULL;
        ThreadPoolFree[i] = NULL;
    }
}

ThreadPoolInterfaceWin::~ThreadPoolInterfaceWin(void)
{
    FreeData();
    myCloseHandle(ghMutexResources);
    if (CSectionOk == TRUE) DeleteCriticalSection(&CriticalSection);
}

bool ThreadPoolInterfaceWin::EnterCS(void)
{
    if ((!Status_Ok) || Error_Occured) return(false);
    else
    {
        EnterCriticalSection(&CriticalSection);
        return(true);
    }
}

void ThreadPoolInterfaceWin::LeaveCS(void)
{
    LeaveCriticalSection(&CriticalSection);
}

bool ThreadPoolInterfaceWin::GetMutex(void)
{
    if ((!Status_Ok) || Error_Occured) return(false);
    else
    {
        WaitForSingleObject(ghMutexResources, INFINITE);
        return(true);
    }
}

void ThreadPoolInterfaceWin::FreeMutex(void)
{
    ReleaseMutex(ghMutexResources);
}

bool ThreadPoolInterfaceWin::CreatePoolEvent(uint8_t num)
{
    if ((!Status_Ok) || (num == 0) || Error_Occured) return(false);

    bool ok = true;
    uint8_t nbrePoolEvent = 0;

    // これまでに作成されたイベント数を数える
    for (uint8_t i = 0; i < MAX_THREAD_POOL; i++)
    {
        if (JobsEnded[i] != NULL) nbrePoolEvent++;
    }

    if (num > nbrePoolEvent)
    {
        uint8_t i = nbrePoolEvent;
        while ((i < num) && ok)
        {
            JobsEnded[i] = CreateEvent(NULL, TRUE, TRUE, NULL);
            ThreadPoolFree[i] = CreateEvent(NULL, TRUE, TRUE, NULL);
            ok = ok && (JobsEnded[i] != NULL) && (ThreadPoolFree[i] != NULL);
            i++;
        }
        if (!ok) Error_Occured = true;
    }
    return(ok);
}

void ThreadPoolInterfaceWin::FreeData(void)
{
    int16_t i;

    if (NbrePool > 0)
    {
        for (i = NbrePool - 1; i >= 0; i--)
            mydelete(ptrPool[i]);
        NbrePool = 0;
    }

    for (i = 0; i < MAX_THREAD_POOL; i++)
    {
        myCloseHandle(ThreadPoolFree[i]);
        myCloseHandle(JobsEnded[i]);
    }

    myCloseHandle(EndExclusive);
}

void ThreadPoolInterfaceWin::FreePool(int8_t nPool)
{
    if (nPool == -1) FreePool();
    else
    {
        if ((nPool >= 0) && (nPool < (int8_t)NbrePool))
        {
            if (ptrPool[nPool] != NULL)
            {
                ThreadPoolWaitFree[nPool] = true;
                while (ThreadPoolRequested[nPool])
                {
                    LeaveCriticalSection(&CriticalSection);
                    WaitForSingleObject(ThreadPoolFree[nPool], INFINITE);
                    EnterCriticalSection(&CriticalSection);
                }
                ptrPool[nPool]->DeAllocateThreads();
                ThreadPoolWaitFree[nPool] = false;
                ThreadPoolUserId[nPool] = 0;
            }

            if (NbreUsers > 0)
            {
                for (uint16_t i = 0; i < NbreUsers; i++)
                {
                    if (TabId[i].nPool == nPool) TabId[i].nPool = -1;
                    TabId[i].nPollTab[nPool] = false;
                }
            }
        }
    }
}

void ThreadPoolInterfaceWin::FreePool(void)
{
    if (NbrePool > 0)
    {
        for (uint16_t i = 0; i < NbrePool; i++)
            ThreadPoolWaitFree[i] = true;
        for (int16_t i = NbrePool - 1; i >= 0; i--)
        {
            if (ptrPool[i] != NULL)
            {
                while (ThreadPoolRequested[i])
                {
                    LeaveCriticalSection(&CriticalSection);
                    WaitForSingleObject(ThreadPoolFree[i], INFINITE);
                    EnterCriticalSection(&CriticalSection);
                }
                ptrPool[i]->DeAllocateThreads();
            }
        }
        for (uint16_t i = 0; i < NbrePool; i++)
        {
            ThreadPoolWaitFree[i] = false;
            ThreadPoolUserId[i] = 0;
        }
    }

    if (NbreUsers > 0)
    {
        for (uint16_t i = 0; i < NbreUsers; i++)
        {
            TabId[i].nPool = -1;
            if (NbrePool > 0)
            {
                for (uint8_t j = 0; j < NbrePool; j++)
                    TabId[i].nPollTab[j] = false;
            }
        }
    }
}

#endif // defined(_WIN32) || defined(_WIN64) 