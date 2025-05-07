#include "ThreadPoolBase.h"

ThreadPoolBase::ThreadPoolBase(void): Status_Ok(true)
{
    int16_t i;

    for (i=0; i<MAX_MT_THREADS; i++)
    {
        MT_Thread[i].MTData = NULL;
        MT_Thread[i].f_process = 0;
        MT_Thread[i].thread_Id = (uint8_t)i;
        ThreadSleep[i] = true;
    }
    TotalThreadsRequested = 0;
    CurrentThreadsAllocated = 0;
    CurrentThreadsUsed = 0;

    Get_CPU_Info(CPU);
    if ((CPU.NbLogicCPU == 0) || (CPU.NbPhysCore == 0)) Status_Ok = false;
}

ThreadPoolBase::~ThreadPoolBase()
{
    DestroyThreadPool();
}

uint8_t ThreadPoolBase::GetThreadNumber(uint8_t thread_number, bool logical)
{
    const uint8_t nCPU = (logical) ? CPU.NbLogicCPU : CPU.NbPhysCore;

    if (thread_number == 0) return((nCPU > MAX_MT_THREADS) ? MAX_MT_THREADS : nCPU);
    else return(thread_number);
}

bool ThreadPoolBase::AllocateThreads(uint8_t thread_number, uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep)
{
    if ((!Status_Ok) || (thread_number == 0)) return(false);

    if (thread_number > CurrentThreadsAllocated)
    {
        TotalThreadsRequested = thread_number;
        CreateThreadPool(offset_core, offset_ht, UseMaxPhysCore, SetAffinity, sleep);
    }

    return(Status_Ok);
}

bool ThreadPoolBase::ChangeThreadsAffinity(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep)
{
    if ((!Status_Ok) || (CurrentThreadsAllocated == 0)) return(false);

    CreateThreadPool(offset_core, offset_ht, UseMaxPhysCore, SetAffinity, sleep);

    return(Status_Ok);
}

bool ThreadPoolBase::DeAllocateThreads(void)
{
    if (!Status_Ok) return(false);

    FreeThreadPool();

    return(true);
}

bool ThreadPoolBase::RequestThreadPool(uint8_t thread_number, Public_MT_Data_Thread *Data)
{
    if ((!Status_Ok) || (thread_number > CurrentThreadsAllocated)) return(false);
    
    for(uint8_t i = 0; i < thread_number; i++)
    {
        MT_Thread[i].MTData = Data + i;
    }
    
    CurrentThreadsUsed = thread_number;

    return(true);    
}

bool ThreadPoolBase::ReleaseThreadPool(bool sleep)
{
    if (!Status_Ok) return(false);

    for(uint8_t i = 0; i < CurrentThreadsUsed; i++)
    {
        MT_Thread[i].MTData = NULL;
    }
    CurrentThreadsUsed = 0;

    return(true);
}

bool ThreadPoolBase::StartThreads(void)
{
    if ((!Status_Ok) || (CurrentThreadsUsed == 0)) return(false);

    for(uint8_t i = 0; i < CurrentThreadsUsed; i++)
    {
        MT_Thread[i].f_process = 1;
    }

    return(true);    
}

bool ThreadPoolBase::WaitThreadsEnd(void)
{
    if ((!Status_Ok) || (CurrentThreadsUsed == 0)) return(false);

    for(uint8_t i = 0; i < CurrentThreadsUsed; i++)
        MT_Thread[i].f_process = 0;

    return(true);
} 