#include "ThreadPoolWin.h"

#if defined(_WIN32) || defined(_WIN64)

#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}

// Helper function to count set bits in the processor mask.
static uint8_t CountSetBits(ULONG_PTR bitMask)
{
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    uint8_t bitSetCount = 0;
    ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;    
    DWORD i;
    
    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}

static ULONG_PTR GetCPUMask(ULONG_PTR bitMask, uint8_t CPU_Nb)
{
    uint8_t LSHIFT = sizeof(ULONG_PTR)*8-1;
    uint8_t i = 0, bitSetCount = 0;
    ULONG_PTR bitTest = 1;    

    CPU_Nb++;
    while (i <= LSHIFT)
    {
        if ((bitMask & bitTest) != 0) bitSetCount++;
        if (bitSetCount == CPU_Nb) return(bitTest);
        else
        {
            i++;
            bitTest <<= 1;
        }
    }
    return(0);
}

static void CreateThreadsMasks(Arch_CPU cpu, ULONG_PTR *TabMask, uint8_t NbThread, uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore)
{
    if (NbThread == 0) return;

    memset(TabMask, 0, NbThread*sizeof(ULONG_PTR));

    if ((cpu.NbLogicCPU == 0) || (cpu.NbPhysCore == 0)) return;

    uint8_t i_cpu = offset_core % cpu.NbPhysCore;
    uint8_t i_ht = offset_ht % cpu.NbHT[i_cpu];
    uint8_t current_thread = 0, nb_cpu = 0;

    if (cpu.NbPhysCore == cpu.NbLogicCPU)
    {
        while (NbThread > current_thread)
        {
            uint8_t Nb_Core_Th = NbThread / cpu.NbPhysCore + (((NbThread % cpu.NbPhysCore) > nb_cpu) ? 1 : 0);

            for(uint8_t i = 0; i < Nb_Core_Th; i++)
                TabMask[current_thread++] = GetCPUMask(cpu.ProcMask[i_cpu], 0);

            nb_cpu++;
            i_cpu = (i_cpu + 1) % cpu.NbPhysCore;
        }
    }
    else
    {
        if (UseMaxPhysCore)
        {
            if (NbThread > cpu.NbPhysCore)
            {
                while (NbThread > current_thread)
                {
                    uint8_t Nb_Core_Th = NbThread / cpu.NbPhysCore + (((NbThread % cpu.NbPhysCore) > nb_cpu) ? 1 : 0);

                    for(uint8_t i = 0; i < Nb_Core_Th; i++)
                        TabMask[current_thread++] = GetCPUMask(cpu.ProcMask[i_cpu], (i + i_ht) % cpu.NbHT[i_cpu]);

                    nb_cpu++;
                    i_cpu = (i_cpu + 1) % cpu.NbPhysCore;
                }
            }
            else
            {
                while (NbThread > current_thread)
                {
                    TabMask[current_thread++] = GetCPUMask(cpu.ProcMask[i_cpu], i_ht);
                    i_cpu = (i_cpu + 1) % cpu.NbPhysCore;
                }
            }
        }
        else
        {
            while (NbThread > current_thread)
            {
                uint8_t Nb_Core_Th = NbThread / cpu.NbPhysCore + (((NbThread % cpu.NbPhysCore) > nb_cpu) ? 1 : 0);

                Nb_Core_Th = (Nb_Core_Th < (cpu.NbHT[i_cpu] - i_ht)) ? (cpu.NbHT[i_cpu] - i_ht) : Nb_Core_Th;
                Nb_Core_Th = (Nb_Core_Th <= (NbThread - current_thread)) ? Nb_Core_Th : (NbThread - current_thread);

                for (uint8_t i = 0; i < Nb_Core_Th; i++)
                    TabMask[current_thread++] = GetCPUMask(cpu.ProcMask[i_cpu], i + i_ht);

                i_cpu = (i_cpu + 1) % cpu.NbPhysCore;
                nb_cpu++;
                i_ht = 0;
            }
        }
    }
}

DWORD WINAPI ThreadPoolWin::StaticThreadpool(LPVOID lpParam)
{
    const MT_Data_Thread_Win *data = (MT_Data_Thread_Win *)lpParam;
    
    while (true)
    {
        WaitForSingleObject(data->nextJob, INFINITE);
        switch(data->f_process)
        {
            case 1:
                if (data->MTData != NULL)
                {
                    data->MTData->thread_Id = data->thread_Id;
                    if (data->MTData->pFunc != NULL) data->MTData->pFunc(data->MTData);
                }
                break;
            case 255: return(0);
            default: break;
        }
        ResetEvent(data->nextJob);
        SetEvent(data->jobFinished);
    }
}

void ThreadPoolWin::Get_CPU_Info(Arch_CPU& cpu)
{
    bool done = false;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
    DWORD returnLength = 0;
    uint8_t logicalProcessorCount = 0;
    uint8_t processorCoreCount = 0;
    DWORD byteOffset = 0;

    cpu.NbLogicCPU = 0;
    cpu.NbPhysCore = 0;
    cpu.FullMask = 0;

    while (!done)
    {
        BOOL rc = GetLogicalProcessorInformation(buffer, &returnLength);

        if (rc == FALSE) 
        {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
            {
                myfree(buffer);
                buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);

                if (buffer == NULL) return;
            } 
            else
            {
                myfree(buffer);
                return;
            }
        } 
        else done = true;
    }

    ptr = buffer;

    while ((byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) <= returnLength) 
    {
        switch (ptr->Relationship) 
        {
            case RelationProcessorCore:
                // A hyperthreaded core supplies more than one logical processor.
                cpu.NbHT[processorCoreCount] = CountSetBits(ptr->ProcessorMask);
                logicalProcessorCount += cpu.NbHT[processorCoreCount];
                cpu.ProcMask[processorCoreCount++] = ptr->ProcessorMask;
                cpu.FullMask |= ptr->ProcessorMask;
                break;
            default: break;
        }
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }
    free(buffer);

    cpu.NbPhysCore = processorCoreCount;
    cpu.NbLogicCPU = logicalProcessorCount;
}

ThreadPoolWin::ThreadPoolWin(void): ThreadPoolBase()
{
    int16_t i;

    for (i = 0; i < MAX_MT_THREADS; i++)
    {
        jobFinished[i] = NULL;
        nextJob[i] = NULL;
        MT_Thread_Win[i].MTData = NULL;
        MT_Thread_Win[i].f_process = 0;
        MT_Thread_Win[i].thread_Id = (uint8_t)i;
        MT_Thread_Win[i].jobFinished = NULL;
        MT_Thread_Win[i].nextJob = NULL;
        thds[i] = NULL;
    }
}

void ThreadPoolWin::FreeThreadPool(void) 
{
    int16_t i;

    if (TotalThreadsRequested > 0)
    {
        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            if (thds[i] != NULL)
            {
                if (ThreadSleep[i]) ResumeThread(thds[i]);
                MT_Thread_Win[i].f_process = 255;
                SetEvent(nextJob[i]);
                WaitForSingleObject(thds[i], INFINITE);
                myCloseHandle(thds[i]);
                MT_Thread_Win[i].f_process = 0;
                MT_Thread_Win[i].MTData = NULL;
                MT_Thread_Win[i].jobFinished = NULL;
                MT_Thread_Win[i].nextJob = NULL;
                ThreadSleep[i] = true;
            }
        }

        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            myCloseHandle(nextJob[i]);
            myCloseHandle(jobFinished[i]);
        }
    }

    TotalThreadsRequested = 0;
    CurrentThreadsAllocated = 0;
    CurrentThreadsUsed = 0;
}

void ThreadPoolWin::DestroyThreadPool(void) 
{
    int16_t i;

    if (TotalThreadsRequested > 0)
    {
        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            if (thds[i] != NULL)
            {
                TerminateThread(thds[i], 0);
                myCloseHandle(thds[i]);
            }
        }

        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            myCloseHandle(nextJob[i]);
            myCloseHandle(jobFinished[i]);
        }
    }
}

ThreadPoolWin::~ThreadPoolWin()
{
    // Base class destructor will call DestroyThreadPool
}

void ThreadPoolWin::CreateThreadPool(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep)
{
    int16_t i;

    for(i = 0; i < CurrentThreadsAllocated; i++)
    {
        SuspendThread(thds[i]);
        ThreadSleep[i] = true;
    }

    CreateThreadsMasks(CPU, ThreadMask, TotalThreadsRequested, offset_core, offset_ht, UseMaxPhysCore);

    for(i = 0; i < CurrentThreadsAllocated; i++)
    {
        if (SetAffinity) SetThreadAffinityMask(thds[i], ThreadMask[i]);
        else SetThreadAffinityMask(thds[i], CPU.FullMask);
        if (!sleep)
        {
            ResumeThread(thds[i]);
            ThreadSleep[i] = false;
        }
    }

    if (CurrentThreadsAllocated == TotalThreadsRequested) return;

    i = CurrentThreadsAllocated;
    while ((i < TotalThreadsRequested) && Status_Ok)
    {
        jobFinished[i] = CreateEvent(NULL, TRUE, TRUE, NULL);
        nextJob[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
        MT_Thread_Win[i].jobFinished = jobFinished[i];
        MT_Thread_Win[i].nextJob = nextJob[i];
        MT_Thread[i].MTData = NULL;
        MT_Thread[i].f_process = 0;
        MT_Thread[i].thread_Id = i;
        
        Status_Ok = Status_Ok && ((MT_Thread_Win[i].jobFinished != NULL) && (MT_Thread_Win[i].nextJob != NULL));
        i++;
    }
    if (!Status_Ok)
    {
        FreeThreadPool();
        return;
    }

    i = CurrentThreadsAllocated;
    while ((i < TotalThreadsRequested) && Status_Ok)
    {
        thds[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)StaticThreadpool, &MT_Thread_Win[i], CREATE_SUSPENDED, &tids[i]);
        Status_Ok = Status_Ok && (thds[i] != NULL);
        if (Status_Ok)
        {
            if (SetAffinity) SetThreadAffinityMask(thds[i], ThreadMask[i]);
            else SetThreadAffinityMask(thds[i], CPU.FullMask);
            if (!sleep)
            {
                ResumeThread(thds[i]);
                ThreadSleep[i] = false;
            }
        }
        i++;
    }

    if (!Status_Ok) FreeThreadPool();
    else CurrentThreadsAllocated = TotalThreadsRequested;
}

#endif // defined(_WIN32) || defined(_WIN64) 