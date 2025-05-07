#include "ThreadPoolInterfaceLinux.h"

#if !defined(_WIN32) && !defined(_WIN64)

#include <signal.h>

#define mydelete(ptr) if (ptr!=NULL) { delete ptr; ptr=NULL;}

ThreadPoolInterfaceLinux::ThreadPoolInterfaceLinux(void): ThreadPoolInterfaceBase(),
    CSectionOk(false)
{
    // クリティカルセクションの初期化
    CSectionOk = (pthread_mutex_init(&criticalSection, NULL) == 0);
    if (CSectionOk)
    {
        if (pthread_mutex_init(&mutexResources, NULL) == 0)
        {
            Status_Ok = true;
        }
        else
        {
            CSectionOk = false;
            pthread_mutex_destroy(&criticalSection);
        }
    }

    // セマフォの初期化
    sem_init(&endExclusive, 0, 0);

    for (uint8_t i = 0; i < MAX_THREAD_POOL; i++)
    {
        sem_init(&jobsEnded[i], 0, 1);
        sem_init(&threadPoolFree[i], 0, 1);
    }
}

ThreadPoolInterfaceLinux::~ThreadPoolInterfaceLinux(void)
{
    FreeData();
    pthread_mutex_destroy(&mutexResources);
    if (CSectionOk) pthread_mutex_destroy(&criticalSection);
    
    // セマフォの破棄
    sem_destroy(&endExclusive);
    
    for (uint8_t i = 0; i < MAX_THREAD_POOL; i++)
    {
        sem_destroy(&jobsEnded[i]);
        sem_destroy(&threadPoolFree[i]);
    }
}

bool ThreadPoolInterfaceLinux::EnterCS(void)
{
    if ((!Status_Ok) || Error_Occured) return(false);
    else
    {
        pthread_mutex_lock(&criticalSection);
        return(true);
    }
}

void ThreadPoolInterfaceLinux::LeaveCS(void)
{
    pthread_mutex_unlock(&criticalSection);
}

bool ThreadPoolInterfaceLinux::GetMutex(void)
{
    if ((!Status_Ok) || Error_Occured) return(false);
    else
    {
        pthread_mutex_lock(&mutexResources);
        return(true);
    }
}

void ThreadPoolInterfaceLinux::FreeMutex(void)
{
    pthread_mutex_unlock(&mutexResources);
}

bool ThreadPoolInterfaceLinux::CreatePoolEvent(uint8_t num)
{
    if ((!Status_Ok) || (num == 0) || Error_Occured) return(false);

    bool ok = true;
    uint8_t nbrePoolEvent = 0;

    // これまでに作成されたセマフォ数を数える
    // Linuxではセマフォを初期化時に全て作成するのでカウントのみ
    for (uint8_t i = 0; i < num; i++)
    {
        // セマフォの値をリセット
        sem_init(&jobsEnded[i], 0, 1);    // シグナル状態に設定
        sem_init(&threadPoolFree[i], 0, 1); // シグナル状態に設定
    }
    
    return(ok);
}

void ThreadPoolInterfaceLinux::FreeData(void)
{
    int16_t i;

    if (NbrePool > 0)
    {
        for (i = NbrePool - 1; i >= 0; i--)
            mydelete(ptrPool[i]);
        NbrePool = 0;
    }

    // Linuxではセマフォはデストラクタで破棄するので
    // ここではセマフォ値のリセットのみ行う
    for (i = 0; i < MAX_THREAD_POOL; i++)
    {
        sem_init(&threadPoolFree[i], 0, 1);  // セマフォをシグナル状態に設定
        sem_init(&jobsEnded[i], 0, 1);      // セマフォをシグナル状態に設定
    }

    sem_init(&endExclusive, 0, 0);
}

void ThreadPoolInterfaceLinux::FreePool(int8_t nPool)
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
                    LeaveCS();
                    // スレッドプールの解放を待つ
                    sem_wait(&threadPoolFree[nPool]);
                    EnterCS();
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

void ThreadPoolInterfaceLinux::FreePool(void)
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
                    LeaveCS();
                    // スレッドプールの解放を待つ
                    sem_wait(&threadPoolFree[i]);
                    EnterCS();
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

#endif // !defined(_WIN32) && !defined(_WIN64) 