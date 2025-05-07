#include "ThreadPoolInterface.h"

#if defined(_WIN32) || defined(_WIN64)
#include "ThreadPoolInterfaceWin.h"
#else
#include "ThreadPoolInterfaceLinux.h"
#endif

ThreadPoolInterfaceBase::ThreadPoolInterfaceBase(void): Status_Ok(false), NbrePool(0), NbreUsers(0), 
    Error_Occured(false), ExclusiveMode(false)
{
    TabId = new UserData[MAX_USERS];
    ThreadPoolRequested = new bool[MAX_THREAD_POOL];
    JobsRunning = new bool[MAX_THREAD_POOL];
    ThreadPoolReleased = new bool[MAX_THREAD_POOL];
    ThreadWaitEnd = new bool[MAX_THREAD_POOL];
    ThreadPoolWaitFree = new bool[MAX_THREAD_POOL];
    ThreadPoolUserId = new uint16_t[MAX_THREAD_POOL];

    for (uint8_t i = 0; i < MAX_THREAD_POOL; i++)
    {
        ThreadPoolRequested[i] = false;
        JobsRunning[i] = false;
        ThreadPoolReleased[i] = false;
        ThreadWaitEnd[i] = false;
        ThreadPoolWaitFree[i] = false;
        ThreadPoolUserId[i] = 0;
        ptrPool[i] = NULL;
    }

    for(uint16_t i = 0; i < MAX_USERS; i++)
    {
        TabId[i].UserId = 0;
        TabId[i].nPool = -1;
        for (uint8_t j = 0; j < MAX_THREAD_POOL; j++)
            TabId[i].nPollTab[j] = false;
    }
}

ThreadPoolInterfaceBase::~ThreadPoolInterfaceBase(void)
{
    FreeData();
    
    delete[] TabId;
    delete[] ThreadPoolRequested;
    delete[] JobsRunning;
    delete[] ThreadPoolReleased;
    delete[] ThreadWaitEnd;
    delete[] ThreadPoolWaitFree;
    delete[] ThreadPoolUserId;
}

uint8_t ThreadPoolInterfaceBase::GetThreadNumber(uint8_t thread_number, bool logical)
{
    if ((!Status_Ok) || (NbrePool == 0)) return(0);
    else return(ptrPool[0]->GetThreadNumber(thread_number, logical));
}

int16_t ThreadPoolInterfaceBase::GetUserIdIndex(uint16_t UserId)
{
    if ((UserId == 0) || (NbreUsers == 0)) return(-1);

    uint16_t i = 0;
    while ((NbreUsers > i) && (TabId[i].UserId != UserId)) i++;

    if (i == NbreUsers) return(-1);
    else return(i);
}

bool ThreadPoolInterfaceBase::GetUserId(uint16_t &UserId)
{
    if ((!Status_Ok) || Error_Occured) return(false);

    EnterCS();
    if ((!Status_Ok) || Error_Occured || ((UserId == 0) && (NbreUsers >= MAX_USERS)))
    {
        LeaveCS();
        return(false);
    }

    bool ret_status = true;

    if (UserId == 0)
    {
        uint16_t user = 1;

        if (NbreUsers > 0)
        {
            bool found = false;

            while (!found && (user <= NbreUsers))
            {
                uint16_t i = 0;
                bool search = true;

                while (search && (i < NbreUsers))
                    search = search && (TabId[i++].UserId != user);
                found = search;
                if (!found) user++;
            }
        }
        NbreUsers++;
        UserId = user;
        TabId[NbreUsers-1].UserId = UserId;
        TabId[NbreUsers-1].nPool = -1;
    }
    else ret_status = (GetUserIdIndex(UserId) != -1);

    LeaveCS();

    return(ret_status);
}

bool ThreadPoolInterfaceBase::RemoveUserId(uint16_t UserId)
{
    if ((!Status_Ok) || (UserId == 0)) return(false);

    EnterCS();

    int16_t index = GetUserIdIndex(UserId);

    if ((!Status_Ok) || (index == -1))
    {
        LeaveCS();
        return(false);
    }

    int8_t nPool = 0;

    while(nPool < (int8_t)NbrePool)
    {
        if (TabId[index].nPollTab[nPool])
        {
            while (ThreadPoolRequested[nPool])
            {
                LeaveCS();
                // ここでスレッドプールの解放を待つ
                GetMutex();
                FreeMutex();
                EnterCS();
                index = GetUserIdIndex(UserId);
                if ((!Status_Ok) || (index == -1))
                {
                    LeaveCS();
                    return(false);
                }
            }
        }
        nPool++;
    }

    if (index < NbreUsers-1)
    {
        for(uint16_t i = (uint16_t)index+1; i < NbreUsers; i++)
            TabId[i-1] = TabId[i];
    }
    NbreUsers--;
    TabId[NbreUsers].UserId = 0;
    TabId[NbreUsers].nPool = -1;
    if (NbrePool > 0)
    {
        for(uint8_t i = 0; i < NbrePool; i++)
            TabId[NbreUsers].nPollTab[i] = false;
    }

    LeaveCS();

    return(true);
}

uint8_t ThreadPoolInterfaceBase::GetLogicalCPUNumber(void)
{
    if (Status_Ok && (NbrePool > 0)) return(ptrPool[0]->GetLogicalCPUNumber());
    else return(0);
}

uint8_t ThreadPoolInterfaceBase::GetPhysicalCoreNumber(void)
{
    if (Status_Ok && (NbrePool > 0)) return(ptrPool[0]->GetPhysicalCoreNumber());
    else return(0);
}

ThreadPoolInterfaceBase* ThreadPoolInterfaceBase::Init(uint8_t num)
{
#if defined(_WIN32) || defined(_WIN64)
    static ThreadPoolInterfaceWin PoolInterface;
#else
    static ThreadPoolInterfaceLinux PoolInterface;
#endif

    if ((num > 0) && PoolInterface.Status_Ok && !PoolInterface.Error_Occured)
    {
        if (PoolInterface.GetMutex())
        {
            if (PoolInterface.Status_Ok && !PoolInterface.Error_Occured)
            {
                if (PoolInterface.EnterCS())
                {
                    if (PoolInterface.Status_Ok && !PoolInterface.Error_Occured)
                    {
                        if (num >= MAX_THREAD_POOL) num = MAX_THREAD_POOL;
                    
                        if (num > PoolInterface.NbrePool)
                        {
                            if (PoolInterface.CreatePoolEvent(num))
                            {
                                bool ok = true;

                                while ((PoolInterface.NbrePool < num) && ok)
                                {
#if defined(_WIN32) || defined(_WIN64)
                                    PoolInterface.ptrPool[PoolInterface.NbrePool] = new ThreadPoolWin();
#else
                                    PoolInterface.ptrPool[PoolInterface.NbrePool] = new ThreadPoolLinux();
#endif
                                    ok = ok && (PoolInterface.ptrPool[PoolInterface.NbrePool] != NULL);
                                    PoolInterface.NbrePool++;
                                }
                                if (!ok)
                                {
                                    PoolInterface.Error_Occured = true;
                                    PoolInterface.FreePool();
                                    PoolInterface.Status_Ok = false;
                                    PoolInterface.FreeData();
                                }
                            }
                            else
                            {
                                PoolInterface.FreePool();
                                PoolInterface.Status_Ok = false;
                                PoolInterface.FreeData();
                            }
                        }
                    }
                    PoolInterface.LeaveCS();
                }
            }
            PoolInterface.FreeMutex();
        }
    }

    return(&PoolInterface);
} 