#ifndef __ThreadPoolWin_H__
#define __ThreadPoolWin_H__

#include "ThreadPoolBase.h"

#if defined(_WIN32) || defined(_WIN64)

typedef struct _MT_Data_Thread_Win
{
    Public_MT_Data_Thread *MTData;
    volatile uint8_t f_process, thread_Id;
    volatile HANDLE nextJob, jobFinished;
} MT_Data_Thread_Win;

class ThreadPoolWin : public ThreadPoolBase
{
public:
    ThreadPoolWin(void);
    virtual ~ThreadPoolWin();

protected:
    virtual void FreeThreadPool(void) override;
    virtual void DestroyThreadPool(void) override;
    virtual void CreateThreadPool(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep) override;
    virtual void Get_CPU_Info(Arch_CPU& cpu) override;

private:
    static DWORD WINAPI StaticThreadpool(LPVOID lpParam);

    MT_Data_Thread_Win MT_Thread_Win[MAX_MT_THREADS];
    HANDLE nextJob[MAX_MT_THREADS], jobFinished[MAX_MT_THREADS];
    HANDLE thds[MAX_MT_THREADS];
    DWORD tids[MAX_MT_THREADS];
    ULONG_PTR ThreadMask[MAX_MT_THREADS];

    ThreadPoolWin(const ThreadPoolWin &other);
    ThreadPoolWin& operator = (const ThreadPoolWin &other);
    bool operator == (const ThreadPoolWin &other) const;
    bool operator != (const ThreadPoolWin &other) const;
};

#endif // defined(_WIN32) || defined(_WIN64)

#endif // __ThreadPoolWin_H__ 