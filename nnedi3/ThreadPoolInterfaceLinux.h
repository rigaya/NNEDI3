#ifndef __ThreadPoolInterfaceLinux_H__
#define __ThreadPoolInterfaceLinux_H__

#include "ThreadPoolInterface.h"

#if !defined(_WIN32) && !defined(_WIN64)

#include <pthread.h>
#include <semaphore.h>

class ThreadPoolInterfaceLinux : public ThreadPoolInterfaceBase
{
public:
    ThreadPoolInterfaceLinux(void);
    virtual ~ThreadPoolInterfaceLinux(void);

    virtual bool EnterCS(void) override;
    virtual void LeaveCS(void) override;
    virtual bool GetMutex(void) override;
    virtual void FreeMutex(void) override;
    virtual bool CreatePoolEvent(uint8_t num) override;
    virtual void FreeData(void) override;
    virtual void FreePool(void) override;
    virtual void FreePool(int8_t nPool) override;

private:
    pthread_mutex_t criticalSection;
    pthread_mutex_t mutexResources;
    bool CSectionOk;
    sem_t jobsEnded[MAX_THREAD_POOL], threadPoolFree[MAX_THREAD_POOL];
    sem_t endExclusive;

    // コピー禁止
    ThreadPoolInterfaceLinux(const ThreadPoolInterfaceLinux &other);
    ThreadPoolInterfaceLinux& operator = (const ThreadPoolInterfaceLinux &other);
    bool operator == (const ThreadPoolInterfaceLinux &other) const;
    bool operator != (const ThreadPoolInterfaceLinux &other) const;
};

#endif // !defined(_WIN32) && !defined(_WIN64)

#endif // __ThreadPoolInterfaceLinux_H__ 