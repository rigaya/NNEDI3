// ThreadPoolDLL.cpp
//

#include "ThreadPool.h"
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}
#else
#include <cstring>
#include <cstdio>
#include <utility>
#include <vector>
#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseEvent(ptr); ptr=NULL;}
#endif


// Helper function to count set bits in the processor mask.
static uint8_t CountSetBits(uintptr_t bitMask)
{
    DWORD LSHIFT = sizeof(uintptr_t)*8 - 1;
    uint8_t bitSetCount = 0;
    uintptr_t bitTest = (uintptr_t)1 << LSHIFT;    
    DWORD i;
    
    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}


#if defined(_WIN32) || defined(_WIN64)
static void Get_CPU_Info(Arch_CPU& cpu)
{
    bool done = false;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer=NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr=NULL;
    DWORD returnLength=0;
    uint8_t logicalProcessorCount=0;
    uint8_t processorCoreCount=0;
    DWORD byteOffset=0;

	cpu.NbLogicCPU=0;
	cpu.NbPhysCore=0;
	cpu.FullMask=0;

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
	            // A hyperthreaded core supplies more than one logical processor.
				cpu.NbHT[processorCoreCount]=CountSetBits(ptr->ProcessorMask);
		        logicalProcessorCount+=cpu.NbHT[processorCoreCount];
				cpu.ProcMask[processorCoreCount++]=ptr->ProcessorMask;
				cpu.FullMask|=ptr->ProcessorMask;
			    break;
			default : break;
        }
        byteOffset+=sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }
	free(buffer);

	cpu.NbPhysCore=processorCoreCount;
	cpu.NbLogicCPU=logicalProcessorCount;
}
#else
static void Get_CPU_Info(Arch_CPU& cpu)
{
    cpu.NbLogicCPU = 0;
    cpu.NbPhysCore = 0;
    cpu.FullMask = 0;

    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp == NULL) return;

    // Linux での処理
    char line[256];
    int physicalId = -1;
    int coreId = -1;
    int processor = -1;
    std::vector<int> uniquePhysicalIds;
    std::vector<std::pair<int, int>> coreInfo; // physical_id, core_id のペア

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strncmp(line, "processor", 9) == 0) {
            sscanf(line, "processor\t: %d", &processor);
        } else if (strncmp(line, "physical id", 11) == 0) {
            sscanf(line, "physical id\t: %d", &physicalId);
        } else if (strncmp(line, "core id", 7) == 0) {
            sscanf(line, "core id\t: %d", &coreId);
        }

        // 空行が来たら1つのCPUの情報が終わり
        if (strlen(line) <= 1) {
            if (processor >= 0 && physicalId >= 0 && coreId >= 0) {
                // 論理プロセッサ数をカウント
                cpu.NbLogicCPU++;

                // ユニークな物理IDを記録
                bool found = false;
                for (size_t i = 0; i < uniquePhysicalIds.size(); i++) {
                    if (uniquePhysicalIds[i] == physicalId) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    uniquePhysicalIds.push_back(physicalId);
                }

                // 物理ID+コアIDのペアを記録
                coreInfo.push_back(std::make_pair(physicalId, coreId));

                // マスクを設定
                uintptr_t processorMask = (uintptr_t)1 << processor;
                cpu.FullMask |= processorMask;

                // 次のCPUのために変数をリセット
                processor = -1;
                physicalId = -1;
                coreId = -1;
            }
        }
    }
    fclose(fp);

    // 物理コア数を計算（ユニークな物理ID+コアIDの組み合わせ）
    std::vector<std::pair<int, int>> uniqueCores;
    for (size_t i = 0; i < coreInfo.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < uniqueCores.size(); j++) {
            if (uniqueCores[j].first == coreInfo[i].first && 
                uniqueCores[j].second == coreInfo[i].second) {
                found = true;
                break;
            }
        }
        if (!found) {
            uniqueCores.push_back(coreInfo[i]);
        }
    }
    cpu.NbPhysCore = uniqueCores.size() > 0 ? (uint8_t)uniqueCores.size() : 1;

    // 各物理コアごとのHT数とマスクを設定
    for (uint8_t i = 0; i < cpu.NbPhysCore && i < 64; i++) {
        if (i < uniqueCores.size()) {
            int physId = uniqueCores[i].first;
            int coreId = uniqueCores[i].second;
            
            // このコアに対応する論理プロセッサをカウント
            uint8_t htCount = 0;
            cpu.ProcMask[i] = 0;
            
            for (uint8_t j = 0; j < coreInfo.size(); j++) {
                if (coreInfo[j].first == physId && coreInfo[j].second == coreId) {
                    htCount++;
                    // プロセッサマスクを論理IDに基づいて設定
                    cpu.ProcMask[i] |= ((uintptr_t)1 << j);
                }
            }
            cpu.NbHT[i] = htCount;
        } else {
            cpu.NbHT[i] = 0;
            cpu.ProcMask[i] = 0;
        }
    }

    // NbPhysCoreが0の場合はフォールバック
    if (cpu.NbPhysCore == 0) {
        cpu.NbPhysCore = 1;
        cpu.NbLogicCPU = cpu.NbLogicCPU > 0 ? cpu.NbLogicCPU : 1;
        cpu.NbHT[0] = cpu.NbLogicCPU;
        cpu.ProcMask[0] = cpu.FullMask;
    }
}
#endif


static uintptr_t GetCPUMask(uintptr_t bitMask, uint8_t CPU_Nb)
{
    uint8_t LSHIFT=sizeof(uintptr_t)*8-1;
    uint8_t i=0,bitSetCount=0;
    uintptr_t bitTest=1;    

	CPU_Nb++;
	while (i<=LSHIFT)
	{
		if ((bitMask & bitTest)!=0) bitSetCount++;
		if (bitSetCount==CPU_Nb) return(bitTest);
		else
		{
			i++;
			bitTest<<=1;
		}
	}
	return(0);
}


static void CreateThreadsMasks(Arch_CPU cpu, uintptr_t *TabMask,uint8_t NbThread,uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore)
{
	if (NbThread==0) return;

	memset(TabMask,0,NbThread*sizeof(uintptr_t));

	if ((cpu.NbLogicCPU==0) || (cpu.NbPhysCore==0)) return;

	uint8_t i_cpu=offset_core%cpu.NbPhysCore;
	uint8_t i_ht=offset_ht%cpu.NbHT[i_cpu];
	uint8_t current_thread=0,nb_cpu=0;

	if (cpu.NbPhysCore==cpu.NbLogicCPU)
	{
		while (NbThread>current_thread)
		{
			uint8_t Nb_Core_Th=NbThread/cpu.NbPhysCore+( ((NbThread%cpu.NbPhysCore)>nb_cpu) ? 1:0 );

			for(uint8_t i=0; i<Nb_Core_Th; i++)
				TabMask[current_thread++]=GetCPUMask(cpu.ProcMask[i_cpu],0);

			nb_cpu++;
			i_cpu=(i_cpu+1)%cpu.NbPhysCore;
		}
	}
	else
	{
		if (UseMaxPhysCore)
		{
			if (NbThread>cpu.NbPhysCore)
			{
				while (NbThread>current_thread)
				{
					uint8_t Nb_Core_Th=NbThread/cpu.NbPhysCore+( ((NbThread%cpu.NbPhysCore)>nb_cpu) ? 1:0 );

					for(uint8_t i=0; i<Nb_Core_Th; i++)
						TabMask[current_thread++]=GetCPUMask(cpu.ProcMask[i_cpu],(i+i_ht)%cpu.NbHT[i_cpu]);

					nb_cpu++;
					i_cpu=(i_cpu+1)%cpu.NbPhysCore;
				}
			}
			else
			{
				while (NbThread>current_thread)
				{
					TabMask[current_thread++]=GetCPUMask(cpu.ProcMask[i_cpu],i_ht);
					i_cpu=(i_cpu+1)%cpu.NbPhysCore;
				}
			}
		}
		else
		{
			while (NbThread>current_thread)
			{
				uint8_t Nb_Core_Th=NbThread/cpu.NbPhysCore+( ((NbThread%cpu.NbPhysCore)>nb_cpu) ? 1:0 );

				Nb_Core_Th=(Nb_Core_Th<(cpu.NbHT[i_cpu]-i_ht)) ? (cpu.NbHT[i_cpu]-i_ht):Nb_Core_Th;
				Nb_Core_Th=(Nb_Core_Th<=(NbThread-current_thread)) ? Nb_Core_Th:(NbThread-current_thread);

				for (uint8_t i=0; i<Nb_Core_Th; i++)
					TabMask[current_thread++]=GetCPUMask(cpu.ProcMask[i_cpu],i+i_ht);

				i_cpu=(i_cpu+1)%cpu.NbPhysCore;
				nb_cpu++;
				i_ht=0;
			}
		}
	}
}


void ThreadPool::ThreadFunction(MT_Data_Thread *data)
{
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
			case 255:
				return;
			default:
				break;
		}
		ResetEvent(data->nextJob);
		SetEvent(data->jobFinished);
	}
}


ThreadPool::ThreadPool(void): MT_Thread(),
  nextJob(),
  jobFinished(),
  threads(),
  ThreadMask(),
  ThreadSleep(),
  Status_Ok(true),
  TotalThreadsRequested(0),
  CurrentThreadsAllocated(0),
  CurrentThreadsUsed(0)
{
	for (int i = 0; i < MAX_MT_THREADS; i++) {
		nextJob.push_back(unique_event(nullptr, nullptr));
		jobFinished.push_back(unique_event(nullptr, nullptr));
	}
	for (int16_t i = 0; i < MAX_MT_THREADS; i++)
	{
		MT_Thread[i].MTData = NULL;
		MT_Thread[i].f_process = 0;
		MT_Thread[i].thread_Id = (uint8_t)i;
		MT_Thread[i].jobFinished = NULL;
		MT_Thread[i].nextJob = NULL;
		ThreadSleep[i] = true;
	}
	TotalThreadsRequested = 0;
	CurrentThreadsAllocated = 0;
	CurrentThreadsUsed = 0;

	Get_CPU_Info(CPU);
	if ((CPU.NbLogicCPU == 0) || (CPU.NbPhysCore == 0)) Status_Ok = false;
}


void ThreadPool::FreeThreadPool(void) 
{
	if (TotalThreadsRequested > 0)
	{
		for (int16_t i = TotalThreadsRequested - 1; i >= 0; i--)
		{
			if (i < threads.size() && threads[i].joinable())
			{
				MT_Thread[i].f_process = 255;
				SetEvent(nextJob[i].get());
				threads[i].join();
				MT_Thread[i].f_process = 0;
				MT_Thread[i].MTData = NULL;
				MT_Thread[i].jobFinished = NULL;
				MT_Thread[i].nextJob = NULL;
				ThreadSleep[i] = true;
			}
		}

		threads.clear();

		for (int16_t i = TotalThreadsRequested - 1; i >= 0; i--)
		{
			nextJob[i].reset();
			jobFinished[i].reset();
		}
	}

	TotalThreadsRequested = 0;
	CurrentThreadsAllocated = 0;
	CurrentThreadsUsed = 0;
}


/*
This function is called by the destructor only, meaning there
is a high probability being in an "unload DLL" stage when this
function is called.
In normal usage, threads should have been exited "properly"
before by a FreeThreadPool call, and this function should
do nothing. But, if unfortunately it's not the case, this
function will clean-up the remaining threads in the "hard" way,
the "proper" way not being possible anymore if we are in
an "unload DLL" stage.
*/

void ThreadPool::DestroyThreadPool(void) 
{
	if (TotalThreadsRequested > 0)
	{
		for (auto& thread : threads)
		{
			if (thread.joinable())
			{
				thread.detach();
			}
		}
		threads.clear();

		for (int16_t i = TotalThreadsRequested - 1; i >= 0; i--)
		{
			nextJob[i].reset();
			jobFinished[i].reset();
		}
	}
}


ThreadPool::~ThreadPool()
{
	DestroyThreadPool();
}


uint8_t ThreadPool::GetThreadNumber(uint8_t thread_number,bool logical)
{
	const uint8_t nCPU=(logical) ? CPU.NbLogicCPU:CPU.NbPhysCore;

	if (thread_number==0) return((nCPU>MAX_MT_THREADS) ? MAX_MT_THREADS:nCPU);
	else return(thread_number);
}


bool ThreadPool::AllocateThreads(uint8_t thread_number,uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep)
{
	if ((!Status_Ok) || (thread_number==0)) return(false);

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
	// 既存のスレッドを停止
	for (size_t i = 0; i < threads.size(); i++)
	{
		ThreadSleep[i] = true;
	}

	CreateThreadsMasks(CPU, ThreadMask, TotalThreadsRequested, offset_core, offset_ht, UseMaxPhysCore);

	// 既存のスレッドのアフィニティを設定
	for (size_t i = 0; i < threads.size(); i++)
	{
		if (SetAffinity)
		{
			SetThreadAffinityMask(threads[i].native_handle(), ThreadMask[i]);
		}
		else
		{
			SetThreadAffinityMask(threads[i].native_handle(), CPU.FullMask);
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
				SetThreadAffinityMask(threads.back().native_handle(), ThreadMask[i]);
			}
			else
			{
				SetThreadAffinityMask(threads.back().native_handle(), CPU.FullMask);
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

