#ifndef __ThreadPoolBase_H__
#define __ThreadPoolBase_H__

#include "rgy_osdep.h"
#include "ThreadPoolDef.h"

typedef struct _MT_Data_Thread
{
    Public_MT_Data_Thread *MTData;
    volatile uint8_t f_process, thread_Id;
} MT_Data_Thread;

typedef struct _Arch_CPU
{
    uint8_t NbPhysCore, NbLogicCPU;
    uint8_t NbHT[64];
    size_t ProcMask[64];
    size_t FullMask;
} Arch_CPU;

class ThreadPoolBase
{
public:
    ThreadPoolBase(void);
    virtual ~ThreadPoolBase();

protected:
    Arch_CPU CPU;

public:
    uint8_t GetThreadNumber(uint8_t thread_number, bool logical);
    bool AllocateThreads(uint8_t thread_number, uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep);
    bool ChangeThreadsAffinity(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep);
    bool DeAllocateThreads(void);
    bool RequestThreadPool(uint8_t thread_number, Public_MT_Data_Thread *Data);
    bool ReleaseThreadPool(bool sleep);
    bool StartThreads(void);
    bool WaitThreadsEnd(void);
    bool GetThreadPoolStatus(void) { return(Status_Ok); }
    uint8_t GetCurrentThreadAllocated(void) { return(CurrentThreadsAllocated); }
    uint8_t GetCurrentThreadUsed(void) { return(CurrentThreadsUsed); }
    uint8_t GetLogicalCPUNumber(void) { return(CPU.NbLogicCPU); }
    uint8_t GetPhysicalCoreNumber(void) { return(CPU.NbPhysCore); }

protected:
    MT_Data_Thread MT_Thread[MAX_MT_THREADS];
    volatile bool ThreadSleep[MAX_MT_THREADS];

    volatile bool Status_Ok;
    volatile uint8_t TotalThreadsRequested, CurrentThreadsAllocated, CurrentThreadsUsed;
    
    virtual void FreeThreadPool(void) = 0;
    virtual void CreateThreadPool(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep) = 0;
    virtual void Get_CPU_Info(Arch_CPU& cpu) { cpu = Arch_CPU(); };

private:
    ThreadPoolBase(const ThreadPoolBase &other);
    ThreadPoolBase& operator = (const ThreadPoolBase &other);
    bool operator == (const ThreadPoolBase &other) const;
    bool operator != (const ThreadPoolBase &other) const;
};

#endif // __ThreadPoolBase_H__ 