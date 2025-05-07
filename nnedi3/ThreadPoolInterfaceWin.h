#ifndef __ThreadPoolInterfaceWin_H__
#define __ThreadPoolInterfaceWin_H__

#include "ThreadPoolInterface.h"

#if defined(_WIN32) || defined(_WIN64)

class ThreadPoolInterfaceWin : public ThreadPoolInterfaceBase
{
public:
    ThreadPoolInterfaceWin(void);
    virtual ~ThreadPoolInterfaceWin(void);

    virtual bool EnterCS(void) override;
    virtual void LeaveCS(void) override;
    virtual bool GetMutex(void) override;
    virtual void FreeMutex(void) override;
    virtual bool CreatePoolEvent(uint8_t num) override;
    virtual void FreeData(void) override;
    virtual void FreePool(void) override;
    virtual void FreePool(int8_t nPool) override;

private:
    CRITICAL_SECTION CriticalSection;
    HANDLE ghMutexResources;
    BOOL CSectionOk;
    HANDLE JobsEnded[MAX_THREAD_POOL], ThreadPoolFree[MAX_THREAD_POOL];
    HANDLE EndExclusive;

    // コピー禁止
    ThreadPoolInterfaceWin(const ThreadPoolInterfaceWin &other);
    ThreadPoolInterfaceWin& operator = (const ThreadPoolInterfaceWin &other);
    bool operator == (const ThreadPoolInterfaceWin &other) const;
    bool operator != (const ThreadPoolInterfaceWin &other) const;
};

#endif // defined(_WIN32) || defined(_WIN64)

#endif // __ThreadPoolInterfaceWin_H__ 