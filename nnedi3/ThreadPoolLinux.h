#ifndef __ThreadPoolLinux_H__
#define __ThreadPoolLinux_H__

#include "ThreadPoolBase.h"

#if !defined(_WIN32) && !defined(_WIN64)

#include <pthread.h>
#include <semaphore.h>

typedef struct _MT_Data_Thread_Linux
{
    Public_MT_Data_Thread *MTData;
    volatile uint8_t f_process, thread_Id;
    volatile sem_t *nextJob, *jobFinished;
} MT_Data_Thread_Linux;

class ThreadPoolLinux : public ThreadPoolBase
{
public:
    ThreadPoolLinux(void);
    virtual ~ThreadPoolLinux();

protected:
    virtual void FreeThreadPool(void) override;
    virtual void DestroyThreadPool(void) override;
    virtual void CreateThreadPool(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep) override;
    virtual void Get_CPU_Info(Arch_CPU& cpu) override;

private:
    static void* StaticThreadpool(void* lpParam);
    
    MT_Data_Thread_Linux MT_Thread_Linux[MAX_MT_THREADS];
    sem_t nextJob[MAX_MT_THREADS], jobFinished[MAX_MT_THREADS];
    pthread_t thds[MAX_MT_THREADS];
    size_t ThreadMask[MAX_MT_THREADS];

    ThreadPoolLinux(const ThreadPoolLinux &other);
    ThreadPoolLinux& operator = (const ThreadPoolLinux &other);
    bool operator == (const ThreadPoolLinux &other) const;
    bool operator != (const ThreadPoolLinux &other) const;
};

#endif // !defined(_WIN32) && !defined(_WIN64)

#endif // __ThreadPoolLinux_H__ 