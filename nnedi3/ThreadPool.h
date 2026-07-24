#ifndef __ThreadPool_H__
#define __ThreadPool_H__

#include "rgy_osdep.h"
#include "KUtil.h"
#include "rgy_event.h"
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "ThreadPoolDef.h"

#define THREADPOOL_VERSION "ThreadPool 1.3.4"

typedef struct _MT_Data_Thread
{
	Public_MT_Data_Thread *MTData;
	volatile uint8_t f_process,thread_Id;
	volatile HANDLE nextJob,jobFinished;
	std::atomic<bool> stop;
} MT_Data_Thread;


typedef struct _Logical_CPU
{
	uint32_t id;
} Logical_CPU;


typedef struct _Arch_CPU
{
	std::vector<std::vector<Logical_CPU>> cores;
	std::vector<Logical_CPU> allowedCPUs;
} Arch_CPU;


class ThreadPool
{
	public :
	ThreadPool(void);
	virtual ~ThreadPool();

	protected :

	Arch_CPU CPU;

	public :

	uint8_t GetThreadNumber(uint8_t thread_number,bool logical);
	bool AllocateThreads(uint8_t thread_number,uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep);
	bool ChangeThreadsAffinity(uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep);
	bool DeAllocateThreads(void);
	bool RequestThreadPool(uint8_t thread_number,Public_MT_Data_Thread *Data);
	bool ReleaseThreadPool(bool sleep);
	bool StartThreads(void);
	bool WaitThreadsEnd(void);
	bool GetThreadPoolStatus(void) {return(Status_Ok);}
	uint8_t GetCurrentThreadAllocated(void) {return(CurrentThreadsAllocated);}
	uint8_t GetCurrentThreadUsed(void) {return(CurrentThreadsUsed);}
	uint8_t GetLogicalCPUNumber(void) {return((uint8_t)std::min(CPU.allowedCPUs.size(), (size_t)MAX_MT_THREADS));}
	uint8_t GetPhysicalCoreNumber(void) {return((uint8_t)std::min(CPU.cores.size(), (size_t)MAX_MT_THREADS));}

	protected :

	MT_Data_Thread MT_Thread[MAX_MT_THREADS];
	std::vector<unique_event> nextJob, jobFinished;
	std::vector<std::thread> threads;
	uint32_t ThreadCPU[MAX_MT_THREADS];
	bool ThreadAffinitySet[MAX_MT_THREADS];
	volatile bool ThreadSleep[MAX_MT_THREADS];

	volatile bool Status_Ok;
	volatile uint8_t TotalThreadsRequested,CurrentThreadsAllocated,CurrentThreadsUsed;
	
	void FreeThreadPool(void);
	void DestroyThreadPool(void);
	void CreateThreadPool(uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep);

	private :

	static void ThreadFunction(MT_Data_Thread *data);

	ThreadPool (const ThreadPool &other);
	ThreadPool& operator = (const ThreadPool &other);
	bool operator == (const ThreadPool &other) const;
	bool operator != (const ThreadPool &other) const;
};

#endif // __ThreadPool_H__
