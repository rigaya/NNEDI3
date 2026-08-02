// ThreadPoolDLL.cpp
//

#include "ThreadPool.h"
#include <algorithm>
#include <climits>
#include <cerrno>
#include <map>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}
#else
#include <cstring>
#include <cstdio>
#include <utility>
#include <vector>
#include <unistd.h>
#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseEvent(ptr); ptr=NULL;}
#endif


#if defined(_WIN32) || defined(_WIN64)
static void Get_CPU_Info(Arch_CPU& cpu)
{
    bool done = false;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer=NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr=NULL;
    DWORD returnLength=0;
    DWORD byteOffset=0;

	cpu = Arch_CPU();

    while (!done)
    {
        BOOL rc=GetLogicalProcessorInformation(buffer, &returnLength);

        if (rc==FALSE) 
        {
            if (GetLastError()==ERROR_INSUFFICIENT_BUFFER) 
            {
                myfree(buffer);
                buffer=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);

                if (buffer==NULL) return;
            } 
            else
			{
				myfree(buffer);
				return;
			}
        } 
        else done=true;
    }

    ptr=buffer;

    while ((byteOffset+sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION))<=returnLength) 
    {
        switch (ptr->Relationship) 
        {
		case RelationProcessorCore :
			{
				std::vector<Logical_CPU> core;
				for (uint32_t bit = 0; bit < sizeof(uintptr_t) * CHAR_BIT; bit++)
				{
					if ((ptr->ProcessorMask & ((uintptr_t)1 << bit)) != 0)
					{
						core.push_back({ bit });
						cpu.allowedCPUs.push_back({ bit });
					}
				}
				if (!core.empty()) cpu.cores.push_back(std::move(core));
			}
			    break;
			default : break;
        }
        byteOffset+=sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }
	free(buffer);

}
#else
typedef struct _Linux_CPU_Record
{
    int processor;
    int physicalId;
    int coreId;
} Linux_CPU_Record;

static std::vector<Logical_CPU> GetAllowedCPUs()
{
    std::vector<Logical_CPU> cpus;
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    size_t capacity = std::max((size_t)128, configured > 0 ? (size_t)configured : (size_t)0);
    while (capacity <= (size_t)1 << 20)
    {
        const size_t setSize = CPU_ALLOC_SIZE(capacity);
        cpu_set_t *set = CPU_ALLOC(capacity);
        if (set == nullptr) break;
        CPU_ZERO_S(setSize, set);
        if (sched_getaffinity(0, setSize, set) == 0)
        {
            for (size_t id = 0; id < capacity; id++)
            {
                if (CPU_ISSET_S(id, setSize, set)) cpus.push_back({ (uint32_t)id });
            }
            CPU_FREE(set);
            break;
        }
        const int error = errno;
        CPU_FREE(set);
        if (error != EINVAL) break;
        capacity *= 2;
    }

    if (cpus.empty())
    {
        const long count = sysconf(_SC_NPROCESSORS_ONLN);
        for (long id = 0; id < count; id++) cpus.push_back({ (uint32_t)id });
    }
    return cpus;
}

static void BuildLinuxCPUTopology(Arch_CPU& cpu, const std::map<int, Linux_CPU_Record>& records)
{
    std::map<std::pair<int, int>, size_t> coreMap;
    for (const auto logical : cpu.allowedCPUs)
    {
        const auto record = records.find((int)logical.id);
        const bool topologyAvailable = record != records.end()
            && record->second.physicalId >= 0 && record->second.coreId >= 0;
        const auto key = topologyAvailable
            ? std::make_pair(record->second.physicalId, record->second.coreId)
            : std::make_pair(INT_MIN, (int)logical.id);
        auto core = coreMap.find(key);
        if (core == coreMap.end())
        {
            const size_t index = cpu.cores.size();
            coreMap.emplace(key, index);
            cpu.cores.push_back({ logical });
        }
        else
        {
            cpu.cores[core->second].push_back(logical);
        }
    }
}

static void Get_CPU_Info(Arch_CPU& cpu)
{
    cpu = Arch_CPU();
    cpu.allowedCPUs = GetAllowedCPUs();
    if (cpu.allowedCPUs.empty()) return;

    FILE* fp = fopen("/proc/cpuinfo", "r");
    std::map<int, Linux_CPU_Record> records;
    if (fp != NULL)
    {
        char line[256];
        Linux_CPU_Record current = { -1, -1, -1 };
        auto commit_record = [&]() {
            if (current.processor >= 0) records[current.processor] = current;
            current = { -1, -1, -1 };
        };

        while (fgets(line, sizeof(line), fp) != NULL)
        {
            if (strncmp(line, "processor", 9) == 0)
                sscanf(line, "processor%*[^:]: %d", &current.processor);
            else if (strncmp(line, "physical id", 11) == 0)
                sscanf(line, "physical id%*[^:]: %d", &current.physicalId);
            else if (strncmp(line, "core id", 7) == 0)
                sscanf(line, "core id%*[^:]: %d", &current.coreId);
            else if (line[0] == '\n' || line[0] == '\r')
                commit_record();
        }
        commit_record();
        fclose(fp);
    }

    BuildLinuxCPUTopology(cpu, records);
}
#endif

static void CreateThreadCPUs(const Arch_CPU& cpu, uint32_t *threadCPU, uint8_t threadCount,
    uint8_t offsetCore, uint8_t offsetHT, bool useMaxPhysCore)
{
	if (threadCount == 0 || cpu.cores.empty()) return;
	std::fill(threadCPU, threadCPU + threadCount, UINT32_MAX);

	size_t coreIndex = offsetCore % cpu.cores.size();
	size_t htOffset = offsetHT % cpu.cores[coreIndex].size();
	size_t currentThread = 0;
	size_t coresVisited = 0;
	const bool noSMT = cpu.cores.size() == cpu.allowedCPUs.size();

	while (currentThread < threadCount)
	{
		const auto& core = cpu.cores[coreIndex];
		size_t threadsForCore = 1;
		if (noSMT || threadCount > cpu.cores.size())
		{
			threadsForCore = threadCount / cpu.cores.size()
				+ ((threadCount % cpu.cores.size()) > coresVisited ? 1 : 0);
		}
		if (!useMaxPhysCore)
		{
			threadsForCore = std::max(threadsForCore, core.size() - htOffset);
		}
		threadsForCore = std::min(threadsForCore, (size_t)threadCount - currentThread);

		for (size_t i = 0; i < threadsForCore; i++)
		{
			threadCPU[currentThread++] = core[(i + htOffset) % core.size()].id;
		}
		coreIndex = (coreIndex + 1) % cpu.cores.size();
		coresVisited++;
		htOffset = useMaxPhysCore ? offsetHT % cpu.cores[coreIndex].size() : 0;
	}
}

static bool SetThreadAffinity(std::thread::native_handle_type thread, const std::vector<Logical_CPU>& cpus)
{
	if (cpus.empty()) return false;
#if defined(_WIN32) || defined(_WIN64)
	uintptr_t mask = 0;
	for (const auto cpu : cpus)
	{
		if (cpu.id < sizeof(mask) * CHAR_BIT) mask |= (uintptr_t)1 << cpu.id;
	}
	return mask != 0 && SetThreadAffinityMask(thread, mask) != 0;
#else
	uint32_t maxCPU = 0;
	for (const auto cpu : cpus) maxCPU = std::max(maxCPU, cpu.id);
	const size_t capacity = std::max((size_t)CPU_SETSIZE, (size_t)maxCPU + 1);
	const size_t setSize = CPU_ALLOC_SIZE(capacity);
	cpu_set_t *set = CPU_ALLOC(capacity);
	if (set == nullptr) return false;
	CPU_ZERO_S(setSize, set);
	for (const auto cpu : cpus) CPU_SET_S(cpu.id, setSize, set);
	const bool result = pthread_setaffinity_np(thread, setSize, set) == 0;
	CPU_FREE(set);
	return result;
#endif
}


void ThreadPool::ThreadFunction(MT_Data_Thread *data)
{
	while (true)
	{
		WaitForSingleObject(data->nextJob, INFINITE);
		if (data->stop.load(std::memory_order_acquire)) return;
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
				return;
			default:
				break;
		}
		ResetEvent(data->nextJob);
		SetEvent(data->jobFinished);
		if (data->stop.load(std::memory_order_acquire)) return;
	}
}


ThreadPool::ThreadPool(void): MT_Thread(),
  nextJob(),
  jobFinished(),
  threads(),
  ThreadCPU(),
  ThreadAffinitySet(),
  ThreadSleep(),
  Status_Ok(true),
  TotalThreadsRequested(0),
  CurrentThreadsAllocated(0),
  CurrentThreadsUsed(0)
{
	for (int i = 0; i < MAX_MT_THREADS; i++) {
		nextJob.push_back(unique_event(nullptr, CloseEvent));
		jobFinished.push_back(unique_event(nullptr, CloseEvent));
	}
	for (int16_t i = 0; i < MAX_MT_THREADS; i++)
	{
		MT_Thread[i].MTData = NULL;
		MT_Thread[i].f_process = 0;
		MT_Thread[i].thread_Id = (uint8_t)i;
		MT_Thread[i].jobFinished = NULL;
		MT_Thread[i].nextJob = NULL;
		MT_Thread[i].stop.store(false, std::memory_order_relaxed);
		ThreadCPU[i] = UINT32_MAX;
		ThreadAffinitySet[i] = false;
		ThreadSleep[i] = true;
	}
	TotalThreadsRequested = 0;
	CurrentThreadsAllocated = 0;
	CurrentThreadsUsed = 0;

	Get_CPU_Info(CPU);
	if (CPU.allowedCPUs.empty() || CPU.cores.empty()) Status_Ok = false;
}


void ThreadPool::FreeThreadPool(void) 
{
	if (TotalThreadsRequested > 0)
	{
		for (size_t i = 0; i < threads.size(); i++)
		{
			MT_Thread[i].stop.store(true, std::memory_order_release);
			SetEvent(nextJob[i].get());
		}

		for (auto& thread : threads)
		{
			if (thread.joinable()) thread.join();
		}

		threads.clear();

		for (int16_t i = TotalThreadsRequested - 1; i >= 0; i--)
		{
			MT_Thread[i].f_process = 0;
			MT_Thread[i].MTData = NULL;
			MT_Thread[i].jobFinished = NULL;
			MT_Thread[i].nextJob = NULL;
			MT_Thread[i].stop.store(false, std::memory_order_relaxed);
			ThreadCPU[i] = UINT32_MAX;
			ThreadAffinitySet[i] = false;
			ThreadSleep[i] = true;
			nextJob[i].reset();
			jobFinished[i].reset();
		}
	}

	TotalThreadsRequested = 0;
	CurrentThreadsAllocated = 0;
	CurrentThreadsUsed = 0;
}


// デストラクタから呼ばれた場合も、ワーカーを停止してから待機資源を解放する。

void ThreadPool::DestroyThreadPool(void) 
{
	FreeThreadPool();
}


ThreadPool::~ThreadPool()
{
	DestroyThreadPool();
}


uint8_t ThreadPool::GetThreadNumber(uint8_t thread_number,bool logical)
{
	const size_t nCPU = logical ? CPU.allowedCPUs.size() : CPU.cores.size();

	if (thread_number==0) return((uint8_t)std::min(nCPU, (size_t)MAX_MT_THREADS));
	else return((uint8_t)std::min((size_t)thread_number, (size_t)MAX_MT_THREADS));
}


bool ThreadPool::AllocateThreads(uint8_t thread_number,uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep)
{
	if ((!Status_Ok) || (thread_number==0) || (thread_number>MAX_MT_THREADS)) return(false);

	if (thread_number>CurrentThreadsAllocated)
	{
		TotalThreadsRequested=thread_number;
		CreateThreadPool(offset_core,offset_ht,UseMaxPhysCore,SetAffinity,sleep);
	}

	return(Status_Ok);
}

bool ThreadPool::ChangeThreadsAffinity(uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep)
{
	if ((!Status_Ok) || (CurrentThreadsAllocated==0)) return(false);

	CreateThreadPool(offset_core,offset_ht,UseMaxPhysCore,SetAffinity,sleep);

	return(Status_Ok);
}

bool ThreadPool::DeAllocateThreads(void)
{
	if (!Status_Ok) return(false);

	FreeThreadPool();

	return(true);
}


void ThreadPool::CreateThreadPool(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep)
{
	(void)sleep;
	// 既存のスレッドを停止
	for (size_t i = 0; i < threads.size(); i++)
	{
		ThreadSleep[i] = true;
	}

	if (SetAffinity)
	{
		CreateThreadCPUs(CPU, ThreadCPU, TotalThreadsRequested, offset_core, offset_ht, UseMaxPhysCore);
	}

	// 既存のスレッドのアフィニティを設定
	for (size_t i = 0; i < threads.size(); i++)
	{
		if (SetAffinity)
		{
			ThreadAffinitySet[i] = ThreadCPU[i] != UINT32_MAX
				&& SetThreadAffinity(threads[i].native_handle(), { { ThreadCPU[i] } });
		}
		else if (ThreadAffinitySet[i])
		{
			SetThreadAffinity(threads[i].native_handle(), CPU.allowedCPUs);
			ThreadAffinitySet[i] = false;
		}
	}

	if (CurrentThreadsAllocated == TotalThreadsRequested) return;

	// 新しいスレッドを作成
	size_t i = CurrentThreadsAllocated;
	while ((i < TotalThreadsRequested) && Status_Ok)
	{
		jobFinished[i] = CreateEventUnique(NULL, TRUE, TRUE);
		nextJob[i] = CreateEventUnique(NULL, TRUE, FALSE);
		MT_Thread[i].jobFinished = jobFinished[i].get();
		MT_Thread[i].nextJob = nextJob[i].get();
		MT_Thread[i].stop.store(false, std::memory_order_relaxed);
		Status_Ok = Status_Ok && (jobFinished[i] && nextJob[i]);
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
		threads.emplace_back(ThreadFunction, &MT_Thread[i]);
		Status_Ok = Status_Ok && threads.back().joinable();
		
		if (Status_Ok)
		{
			if (SetAffinity)
			{
				ThreadAffinitySet[i] = ThreadCPU[i] != UINT32_MAX
					&& SetThreadAffinity(threads.back().native_handle(), { { ThreadCPU[i] } });
			}
		}
		i++;
	}

	if (!Status_Ok)
	{
		FreeThreadPool();
	}
	else
	{
		CurrentThreadsAllocated = TotalThreadsRequested;
	}
}


bool ThreadPool::RequestThreadPool(uint8_t thread_number, Public_MT_Data_Thread *Data)
{
	if ((!Status_Ok) || (thread_number > CurrentThreadsAllocated)) return(false);
	
	for(uint8_t i = 0; i < thread_number; i++)
	{
		MT_Thread[i].MTData = Data + i;
		ThreadSleep[i] = false;
	}
	
	CurrentThreadsUsed = thread_number;

	return(true);	
}


bool ThreadPool::ReleaseThreadPool(bool sleep)
{
	if (!Status_Ok) return(false);

	for(uint8_t i = 0; i < CurrentThreadsUsed; i++)
	{
		if (sleep)
		{
			ThreadSleep[i] = true;
		}
		MT_Thread[i].MTData = NULL;
	}
	CurrentThreadsUsed = 0;

	return(true);
}


bool ThreadPool::StartThreads(void)
{
	if ((!Status_Ok) || (CurrentThreadsUsed == 0)) return(false);

	for(uint8_t i = 0; i < CurrentThreadsUsed; i++)
	{
		MT_Thread[i].f_process = 1;
		ResetEvent(jobFinished[i].get());
		SetEvent(nextJob[i].get());
	}

	return(true);	
}


bool ThreadPool::WaitThreadsEnd(void)
{
	if ((!Status_Ok) || (CurrentThreadsUsed == 0)) return(false);

	HANDLE handles[MAX_MT_THREADS];
	for (uint8_t i = 0; i < CurrentThreadsUsed; i++) {
		handles[i] = jobFinished[i].get();
	}
	WaitForMultipleObjects(CurrentThreadsUsed, handles, TRUE, INFINITE);

	for(uint8_t i = 0; i < CurrentThreadsUsed; i++)
		MT_Thread[i].f_process = 0;

	return(true);
}
