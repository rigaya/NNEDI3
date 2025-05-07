#include "ThreadPoolLinux.h"

#if !defined(_WIN32) && !defined(_WIN64)

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

// Helper function to count set bits in the processor mask.
static uint8_t CountSetBits(size_t bitMask)
{
    uint8_t bitSetCount = 0;
    uint8_t LSHIFT = sizeof(size_t) * 8 - 1;
    size_t bitTest = (size_t)1 << LSHIFT;    
    
    for (uint8_t i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest) ? 1 : 0);
        bitTest /= 2;
    }

    return bitSetCount;
}

static size_t GetCPUMask(size_t bitMask, uint8_t CPU_Nb)
{
    uint8_t LSHIFT = sizeof(size_t) * 8 - 1;
    uint8_t i = 0, bitSetCount = 0;
    size_t bitTest = 1;

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

static void CreateThreadsMasks(Arch_CPU cpu, size_t *TabMask, uint8_t NbThread, uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore)
{
    if (NbThread == 0) return;

    memset(TabMask, 0, NbThread * sizeof(size_t));

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

void* ThreadPoolLinux::StaticThreadpool(void* lpParam)
{
    const MT_Data_Thread_Linux *data = (MT_Data_Thread_Linux *)lpParam;
    
    while (true)
    {
        sem_wait((sem_t*)data->nextJob);
        switch(data->f_process)
        {
            case 1:
                if (data->MTData != NULL)
                {
                    data->MTData->thread_Id = data->thread_Id;
                    if (data->MTData->pFunc != NULL) data->MTData->pFunc(data->MTData);
                }
                break;
            case 255: 
                sem_post((sem_t*)data->jobFinished);
                return NULL;
            default: break;
        }
        sem_post((sem_t*)data->jobFinished);
        sem_wait((sem_t*)data->nextJob); // Reset semaphore to 0
    }
    
    return NULL;
}

void ThreadPoolLinux::Get_CPU_Info(Arch_CPU& cpu)
{
    // Initialize CPU info
    cpu.NbLogicCPU = 0;
    cpu.NbPhysCore = 0;
    cpu.FullMask = 0;

    // Get number of processors
    cpu.NbLogicCPU = sysconf(_SC_NPROCESSORS_ONLN);
    
    // For Linux, we'll simplify and assume each logical CPU is a physical core
    // TODO: Improve this by parsing /proc/cpuinfo or other methods to get accurate topology
    cpu.NbPhysCore = cpu.NbLogicCPU;
    
    // Create masks for each core
    for (uint8_t i = 0; i < cpu.NbPhysCore; i++) {
        cpu.NbHT[i] = 1;  // Set to 1 by default (no hyperthreading)
        cpu.ProcMask[i] = 1ULL << i;
        cpu.FullMask |= cpu.ProcMask[i];
    }
}

ThreadPoolLinux::ThreadPoolLinux(void): ThreadPoolBase()
{
    int16_t i;

    for (i = 0; i < MAX_MT_THREADS; i++)
    {
        MT_Thread_Linux[i].MTData = NULL;
        MT_Thread_Linux[i].f_process = 0;
        MT_Thread_Linux[i].thread_Id = (uint8_t)i;
        MT_Thread_Linux[i].nextJob = NULL;
        MT_Thread_Linux[i].jobFinished = NULL;
        sem_init(&nextJob[i], 0, 0);       // Initialize to 0
        sem_init(&jobFinished[i], 0, 1);   // Initialize to 1
    }
}

void ThreadPoolLinux::FreeThreadPool(void) 
{
    int16_t i;

    if (TotalThreadsRequested > 0)
    {
        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            if (thds[i] != 0)
            {
                if (ThreadSleep[i]) {
                    // Resume thread if it's sleeping
                    pthread_kill(thds[i], SIGCONT);
                }
                MT_Thread_Linux[i].f_process = 255;
                sem_post(&nextJob[i]);  // Signal thread to exit
                pthread_join(thds[i], NULL);
                
                MT_Thread_Linux[i].f_process = 0;
                MT_Thread_Linux[i].MTData = NULL;
                MT_Thread_Linux[i].jobFinished = NULL;
                MT_Thread_Linux[i].nextJob = NULL;
                ThreadSleep[i] = true;
                thds[i] = 0;
            }
        }

        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            sem_destroy(&nextJob[i]);
            sem_destroy(&jobFinished[i]);
        }
    }

    TotalThreadsRequested = 0;
    CurrentThreadsAllocated = 0;
    CurrentThreadsUsed = 0;
}

void ThreadPoolLinux::DestroyThreadPool(void) 
{
    int16_t i;

    if (TotalThreadsRequested > 0)
    {
        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            if (thds[i] != 0)
            {
                pthread_cancel(thds[i]);
                thds[i] = 0;
            }
        }

        for (i = TotalThreadsRequested - 1; i >= 0; i--)
        {
            sem_destroy(&nextJob[i]);
            sem_destroy(&jobFinished[i]);
        }
    }
}

ThreadPoolLinux::~ThreadPoolLinux()
{
    // Base class destructor will call DestroyThreadPool
}

void ThreadPoolLinux::CreateThreadPool(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep)
{
    int16_t i;

    for(i = 0; i < CurrentThreadsAllocated; i++)
    {
        // Pause thread by sending SIGSTOP
        if (!ThreadSleep[i]) {
            pthread_kill(thds[i], SIGSTOP);
            ThreadSleep[i] = true;
        }
    }

    CreateThreadsMasks(CPU, ThreadMask, TotalThreadsRequested, offset_core, offset_ht, UseMaxPhysCore);

    for(i = 0; i < CurrentThreadsAllocated; i++)
    {
        if (SetAffinity) SetThreadAffinityMask(thds[i], ThreadMask[i]);
        else SetThreadAffinityMask(thds[i], CPU.FullMask);
        
        if (!sleep)
        {
            pthread_kill(thds[i], SIGCONT);
            ThreadSleep[i] = false;
        }
    }

    if (CurrentThreadsAllocated == TotalThreadsRequested) return;

    i = CurrentThreadsAllocated;
    while ((i < TotalThreadsRequested) && Status_Ok)
    {
        sem_init(&jobFinished[i], 0, 1);  // Initialize to 1 (ready)
        sem_init(&nextJob[i], 0, 0);      // Initialize to 0 (wait)
        
        MT_Thread_Linux[i].jobFinished = &jobFinished[i];
        MT_Thread_Linux[i].nextJob = &nextJob[i];
        MT_Thread[i].MTData = NULL;
        MT_Thread[i].f_process = 0;
        MT_Thread[i].thread_Id = i;
        
        Status_Ok = Status_Ok && true;  // Always succeed for Linux
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
        int result = pthread_create(&thds[i], NULL, StaticThreadpool, &MT_Thread_Linux[i]);
        Status_Ok = Status_Ok && (result == 0);
        
        if (Status_Ok)
        {
            if (SetAffinity) SetThreadAffinityMask(thds[i], ThreadMask[i]);
            else SetThreadAffinityMask(thds[i], CPU.FullMask);
            
            if (sleep)
            {
                pthread_kill(thds[i], SIGSTOP);
                ThreadSleep[i] = true;
            }
            else
            {
                ThreadSleep[i] = false;
            }
        }
        i++;
    }

    if (!Status_Ok) FreeThreadPool();
    else CurrentThreadsAllocated = TotalThreadsRequested;
}

#endif // !defined(_WIN32) && !defined(_WIN64) 