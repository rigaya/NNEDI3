// ThreadPoolDLL.cpp: d馭init les fonctions export馥s pour l'application DLL.
//

#include "ThreadPoolInterface.h"
#include "ThreadPool.h"

#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr.get()!=nullptr) { ptr.reset(); }
#define mydelete(ptr) if (ptr!=NULL) { delete ptr; ptr=NULL;}


static ThreadPool *ptrPool[MAX_THREAD_POOL];


ThreadPoolInterface* ThreadPoolInterface::Init(uint8_t num)
{

	static ThreadPoolInterface PoolInterface;

	if ((num>0) && PoolInterface.Status_Ok && !PoolInterface.Error_Occured)
	{
		if (PoolInterface.GetMutex())
		{
			if (PoolInterface.Status_Ok && !PoolInterface.Error_Occured)
			{
				if (PoolInterface.EnterCS())
				{
					if (PoolInterface.Status_Ok && !PoolInterface.Error_Occured)
					{
						if (num>=MAX_THREAD_POOL) num=MAX_THREAD_POOL;
					
						if (num>PoolInterface.NbrePool)
						{
							if (PoolInterface.CreatePoolEvent(num))
							{
								bool ok=true;

								while ((PoolInterface.NbrePool<num) && ok)
								{
									ptrPool[PoolInterface.NbrePool]= new ThreadPool();
									ok=ok && (ptrPool[PoolInterface.NbrePool]!=NULL);
									PoolInterface.NbrePool++;
								}
								if (!ok)
								{
									PoolInterface.Error_Occured=true;
									PoolInterface.FreePool();
									PoolInterface.Status_Ok=false;
									PoolInterface.FreeData();
								}
							}
							else
							{
								PoolInterface.FreePool();
								PoolInterface.Status_Ok=false;
								PoolInterface.FreeData();
							}
						}
					}
					PoolInterface.LeaveCS();
				}
			}
			PoolInterface.FreeMutex();
		}
	}

	return(&PoolInterface);
}


bool ThreadPoolInterface::EnterCS(void)
{
	if ((!Status_Ok) || Error_Occured) return(false);
	else
	{
		CriticalSection.lock();
		return(true);
	}
}


void ThreadPoolInterface::LeaveCS(void)
{
	CriticalSection.unlock();
}


bool ThreadPoolInterface::GetMutex(void)
{
	if ((!Status_Ok) || Error_Occured) return(false);
	else
	{
		ghMutexResources.lock();
		return(true);
	}
}


void ThreadPoolInterface::FreeMutex(void)
{
	ghMutexResources.unlock();
}


bool ThreadPoolInterface::CreatePoolEvent(uint8_t num)
{
	if ((!Status_Ok) || (num==0) || Error_Occured)  return(false);

	bool ok=true;

	if (num>NbrePoolEvent)
	{
		while((NbrePoolEvent<num) && ok)
		{
			JobsEnded[NbrePoolEvent]=CreateEventUnique(NULL,TRUE,TRUE);
			ThreadPoolFree[NbrePoolEvent]=CreateEventUnique(NULL,TRUE,TRUE);
			ok=ok && (JobsEnded[NbrePoolEvent]!=NULL) && (ThreadPoolFree[NbrePoolEvent]!=NULL);
			NbrePoolEvent++;
		}
		if (!ok) Error_Occured=true;
	}
	return(ok);
}


void ThreadPoolInterface::FreeData(void)
{
	int16_t i;

	if (NbrePool>0)
	{
		for(i=NbrePool-1; i>=0; i--)
			mydelete(ptrPool[i]);
		NbrePool=0;
	}

	if (NbrePoolEvent>0)
	{
		for(i=NbrePoolEvent-1; i>=0; i--)
		{
			ThreadPoolFree[i].reset();
			JobsEnded[i].reset();
		}
		NbrePoolEvent=0;
	}

	EndExclusive.reset();
}


void ThreadPoolInterface::FreePool(int8_t nPool)
{
	if (nPool==-1) FreePool();
	else
	{
		if ((nPool>=0) && (nPool<(int8_t)NbrePool))
		{
			if (ptrPool[nPool]!=NULL)
			{
				ThreadPoolWaitFree[nPool]=true;
				while(ThreadPoolRequested[nPool])
				{
					LeaveCS();
					WaitForSingleObject(ThreadPoolFree[nPool].get(), INFINITE);
					EnterCS();
				}
				ptrPool[nPool]->DeAllocateThreads();
				ThreadPoolWaitFree[nPool]=false;
				ThreadPoolUserId[nPool]=0;
			}

			if (NbreUsers>0)
			{
				for(uint16_t i=0; i<NbreUsers; i++)
				{
					if (TabId[i].nPool==nPool) TabId[i].nPool=-1;
					TabId[i].nPollTab[nPool]=false;
				}
			}
		}
	}
}


void ThreadPoolInterface::FreePool(void)
{
	if (NbrePool>0)
	{
		for(uint16_t i=0; i<NbrePool; i++)
			ThreadPoolWaitFree[i]=true;
		for(int16_t i=NbrePool-1; i>=0; i--)
		{
			if (ptrPool[i]!=NULL)
			{
				while(ThreadPoolRequested[i])
				{
					LeaveCS();
					WaitForSingleObject(ThreadPoolFree[i].get(), INFINITE);
					EnterCS();
				}
				ptrPool[i]->DeAllocateThreads();
			}
		}
		for(uint16_t i=0; i<NbrePool; i++)
		{
			ThreadPoolWaitFree[i]=false;
			ThreadPoolUserId[i]=0;
		}
	}

	if (NbreUsers>0)
	{
		for(uint16_t i=0; i<NbreUsers; i++)
		{
			TabId[i].nPool=-1;
			if (NbrePool>0)
			{
				for(uint8_t j=0; j<NbrePool; j++)
					TabId[i].nPollTab[j]=false;
			}
		}
	}
}



ThreadPoolInterface::ThreadPoolInterface(void):
	CSectionOk(false), Status_Ok(false), Error_Occured(false),
	NbrePool(0), NbrePoolEvent(0), JobsEnded(), ThreadPoolFree(),
	EndExclusive(unique_event(nullptr, nullptr))
{
	CSectionOk = true;
	if (CSectionOk)
	{
		// std::mutexは例外的な状況を除いて常に構築できる
		Status_Ok = true;
	}

	ExclusiveMode = false;

	for (uint8_t i=0; i<MAX_THREAD_POOL; i++) {
		JobsEnded.push_back(unique_event(nullptr, nullptr));
		ThreadPoolFree.push_back(unique_event(nullptr, nullptr));
	}

	for (uint8_t i=0; i<MAX_THREAD_POOL; i++)
	{
		ThreadPoolRequested[i]=false;
		ThreadPoolReleased[i]=false;
		ThreadWaitEnd[i]=false;
		ThreadPoolWaitFree[i]=false;
		ThreadPoolUserId[i]=0;
		JobsRunning[i]=false;
		ptrPool[i]=NULL;
	}

	for(uint16_t i=0; i<MAX_USERS; i++)
	{
		TabId[i].UserId=0;
		TabId[i].nPool=-1;
		for (uint8_t j=0; j<MAX_THREAD_POOL; j++)
			TabId[i].nPollTab[j]=false;
	}
}


ThreadPoolInterface::~ThreadPoolInterface(void)
{
	FreeData();
}


uint8_t ThreadPoolInterface::GetThreadNumber(uint8_t thread_number,bool logical)
{
	if ((!Status_Ok) || (NbrePool==0)) return(0);
	else return(ptrPool[0]->GetThreadNumber(thread_number,logical));
}


bool ThreadPoolInterface::CreatePool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured || (num==0) || (num>MAX_THREAD_POOL)) return(false);
	
	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured) 
	{
		ghMutexResources.unlock();
		return(false);
	}
	
	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}

	if (num<=NbrePool)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(true);
	}
	
	if (CreatePoolEvent(num))
	{
		bool ok=true;

		while ((NbrePool<num) && ok)
		{
			ptrPool[NbrePool]= new ThreadPool();
			ok=ok && (ptrPool[NbrePool]!=NULL);
			NbrePool++;
		}
		if (!ok)
		{
			Error_Occured=true;
			FreePool();
			Status_Ok=false;
			FreeData();
		}
	}
	else
	{
		Error_Occured=true;
		FreePool();
		Status_Ok=false;
		FreeData();
	}
	
	CriticalSection.unlock();
	ghMutexResources.unlock();
	
	return(Status_Ok);
}



int16_t ThreadPoolInterface::AddPool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured || (num==0)) return(-1);
	
	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured || ((NbrePool+num)>=MAX_THREAD_POOL)) 
	{
		ghMutexResources.unlock();
		return(-1);
	}
	
	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(-1);
	}
	
	int16_t CurrentPool=(int16_t)NbrePool;
	uint8_t numP=NbrePool+num;
	
	if (CreatePoolEvent(numP))
	{
		bool ok=true;

		while ((NbrePool<numP) && ok)
		{
			ptrPool[NbrePool]= new ThreadPool();
			ok=ok && (ptrPool[NbrePool]!=NULL);
			NbrePool++;
		}
		if (!ok)
		{
			Error_Occured=true;
			FreePool();
			Status_Ok=false;
			FreeData();
			CurrentPool=-1;
		}
	}
	else
	{
		Error_Occured=true;
		FreePool();
		Status_Ok=false;
		FreeData();
		CurrentPool=-1;
	}
	
	CriticalSection.unlock();
	ghMutexResources.unlock();
	
	return(CurrentPool);
}


bool ThreadPoolInterface::DeletePool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured || (num==0)) return(false);
	
	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured || (num>NbrePool))
	{
		ghMutexResources.unlock();
		return(false);
	}
	
	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}
	
	uint8_t CurrentPool=NbrePool;	
	int16_t CurrentNum=(int16_t)CurrentPool-1;
	
	for(uint8_t i=0; i<num; i++)
	{
		if (ptrPool[CurrentNum]!=NULL)
		{
			ThreadPoolWaitFree[CurrentNum]=true;
			while(ThreadPoolRequested[CurrentNum])
			{
				CriticalSection.unlock();
				WaitForSingleObject(ThreadPoolFree[CurrentNum].get(),INFINITE);
				CriticalSection.lock();
			}
			ptrPool[CurrentNum]->DeAllocateThreads();
			if (NbreUsers>0)
			{
				for (uint16_t j=0; j<NbreUsers; j++)
				{
					if (TabId[j].nPool==(int8_t)CurrentNum) TabId[j].nPool=-1;
					TabId[j].nPollTab[CurrentNum]=false;
				}
			}
			ThreadPoolUserId[CurrentNum]=0;
			ThreadPoolWaitFree[CurrentNum--]=false;
			NbrePool--;
		}
	}
	
	CurrentNum=(int16_t)CurrentPool-1;
	for(uint8_t i=0; i<num; i++)
		mydelete(ptrPool[CurrentNum--]);
	
	CurrentNum=(int16_t)CurrentPool-1;
	for(uint8_t i=0; i<num; i++)
	{
		ThreadPoolFree[CurrentNum].reset();
		JobsEnded[CurrentNum--].reset();
	}
	
	CriticalSection.unlock();
	ghMutexResources.unlock();
	
	return(true);
}


bool ThreadPoolInterface::RemovePool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured) return(false);
	
	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured || (num>=NbrePool))
	{
		ghMutexResources.unlock();
		return(false);
	}
	
	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}
	
	if (ptrPool[num]!=NULL)
	{
		ThreadPoolWaitFree[num]=true;
		while(ThreadPoolRequested[num])
		{
			CriticalSection.unlock();
			WaitForSingleObject(ThreadPoolFree[num].get(),INFINITE);
			CriticalSection.lock();
		}
		ptrPool[num]->DeAllocateThreads();
		ThreadPoolUserId[num]=0;
		ThreadPoolWaitFree[num]=false;
	}

	mydelete(ptrPool[num]);
	ThreadPoolFree[num].reset();
	JobsEnded[num].reset();
	
	if (num<(NbrePool-1))
	{
		for(uint8_t i=num; i<NbrePool-1; i++)
		{
			ptrPool[i]=ptrPool[i+1];
			ThreadPoolFree[i]=std::move(ThreadPoolFree[i+1]);
			JobsEnded[i]=std::move(JobsEnded[i+1]);
			ThreadPoolRequested[i]=ThreadPoolRequested[i+1];
			JobsRunning[i]=JobsRunning[i+1];
			ThreadPoolReleased[i]=ThreadPoolReleased[i+1];
			ThreadWaitEnd[i]=ThreadWaitEnd[i+1];
			ThreadPoolUserId[i]=ThreadPoolUserId[i+1];
			ThreadPoolWaitFree[i]=ThreadPoolWaitFree[i+1];
		}
	}
	
	if (NbreUsers>0)
	{
		for (uint16_t i=0; i<NbreUsers; i++)
		{
			if (TabId[i].nPool==(int8_t)num) TabId[i].nPool=-1;
			else
			{
				if (TabId[i].nPool>(int8_t)num) TabId[i].nPool--;			
			}
			TabId[i].nPollTab[num]=false;
			if (num<(NbrePool-1))
			{
				for(uint8_t j=num; j<NbrePool-1; j++)
					TabId[i].nPollTab[j]=TabId[i].nPollTab[j+1];
			}
		}
	}
	
	NbrePool--;
	
	CriticalSection.unlock();
	ghMutexResources.unlock();
	
	return(true);
}


bool ThreadPoolInterface::AllocateThreads(uint8_t thread_number,uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep,int8_t nPool)
{
	if ((!Status_Ok) || Error_Occured || (thread_number==0) || (nPool<-1)) return(false);

	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured  || (NbrePool==0) || (nPool>=(int8_t)NbrePool))
	{
		ghMutexResources.unlock();
		return(false);
	}

	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}

	if (nPool==-1)
	{
		for(uint8_t i=0; i<NbrePool; i++)
		{
			if (thread_number>ptrPool[i]->GetCurrentThreadAllocated())
			{
				while (JobsRunning[i])
				{
					CriticalSection.unlock();
					WaitForSingleObject(JobsEnded[i].get(),INFINITE);
					CriticalSection.lock();
					if ((!Status_Ok) || Error_Occured)
					{
						CriticalSection.unlock();
						ghMutexResources.unlock();
						return(false);
					}
				}
				bool ok=ptrPool[i]->AllocateThreads(thread_number,offset_core,offset_ht,UseMaxPhysCore,SetAffinity,sleep);
				if (!ok)
				{
					Error_Occured=true;
					FreePool();
					Status_Ok=false;
					FreeData();
					CriticalSection.unlock();
					ghMutexResources.unlock();
					return(false);
				}
			}
		}
	}
	else
	{
		if (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated())
		{
			while (JobsRunning[nPool])
			{
				CriticalSection.unlock();
				WaitForSingleObject(JobsEnded[nPool].get(),INFINITE);
				CriticalSection.lock();
				if ((!Status_Ok) || Error_Occured)
				{
					CriticalSection.unlock();
					ghMutexResources.unlock();
					return(false);
				}
			}
			bool ok=ptrPool[nPool]->AllocateThreads(thread_number,offset_core,offset_ht,UseMaxPhysCore,SetAffinity,sleep);
			if (!ok)
			{
				Error_Occured=true;
				FreePool();
				Status_Ok=false;
				FreeData();
				CriticalSection.unlock();
				ghMutexResources.unlock();
				return(false);
			}
		}
	}

	CriticalSection.unlock();
	ghMutexResources.unlock();

	return(true);
}


inline int16_t ThreadPoolInterface::GetUserIdIndex(uint16_t UserId)
{
	if ((UserId==0) || (NbreUsers==0)) return(-1);

	uint16_t i=0;
	while ((NbreUsers>i) && (TabId[i].UserId!=UserId)) i++;

	if (i==NbreUsers) return(-1);
	else return(i);
}


bool ThreadPoolInterface::GetUserId(uint16_t &UserId)
{
	if ((!Status_Ok) || Error_Occured) return(false);

	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured || ((UserId==0) && (NbreUsers>=MAX_USERS)))
	{
		CriticalSection.unlock();
		return(false);
	}

	bool ret_status=true;

	if (UserId==0)
	{
		uint16_t user=1;

		if (NbreUsers>0)
		{
			bool found=false;

			while (!found && (user<=NbreUsers))
			{
				uint16_t i=0;
				bool search=true;

				while (search && (i<NbreUsers))
					search=search && (TabId[i++].UserId!=user);
				found=search;
				if (!found) user++;
			}
		}
		NbreUsers++;
		UserId=user;
		TabId[NbreUsers-1].UserId=UserId;
		TabId[NbreUsers-1].nPool=-1;
	}
	else ret_status=(GetUserIdIndex(UserId)!=-1);

	CriticalSection.unlock();

	return(ret_status);
}


bool ThreadPoolInterface::ChangeThreadsAffinity(uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,bool sleep,int8_t nPool)
{
	if ((!Status_Ok) || Error_Occured || (nPool<-1)) return(false);

	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured || (nPool>=(int8_t)NbrePool))
	{
		ghMutexResources.unlock();
		return(false);
	}

	CriticalSection.lock();
	if ((!Status_Ok) || Error_Occured )
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}

	if (nPool==-1)
	{
		for(uint8_t i=0; i<NbrePool; i++)
		{
			if (ptrPool[i]->GetCurrentThreadAllocated()>0)
			{
				while (JobsRunning[i])
				{
					CriticalSection.unlock();
					WaitForSingleObject(JobsEnded[i].get(),INFINITE);
					CriticalSection.lock();
					if ((!Status_Ok) || Error_Occured)
					{
						CriticalSection.unlock();
						ghMutexResources.unlock();
						return(false);
					}
				}
				bool ok=ptrPool[i]->ChangeThreadsAffinity(offset_core,offset_ht,UseMaxPhysCore,SetAffinity,sleep);
				if (!ok)
				{
					Error_Occured=true;
					FreePool();
					Status_Ok=false;
					FreeData();
					CriticalSection.unlock();
					ghMutexResources.unlock();
					return(false);
				}
			}
		}
	}
	else
	{
		if (ptrPool[nPool]->GetCurrentThreadAllocated()>0)
		{
			while (JobsRunning[nPool])
			{
				CriticalSection.unlock();
				WaitForSingleObject(JobsEnded[nPool].get(),INFINITE);
				CriticalSection.lock();
				if ((!Status_Ok) || Error_Occured)
				{
					CriticalSection.unlock();
					ghMutexResources.unlock();
					return(false);
				}
			}
			bool ok=ptrPool[nPool]->ChangeThreadsAffinity(offset_core,offset_ht,UseMaxPhysCore,SetAffinity,sleep);
			if (!ok)
			{
				Error_Occured=true;
				FreePool();
				Status_Ok=false;
				FreeData();
				CriticalSection.unlock();
				ghMutexResources.unlock();
				return(false);
			}
		}
	}

	CriticalSection.unlock();
	ghMutexResources.unlock();

	return(true);
}


bool ThreadPoolInterface::DeAllocatePoolThreads(uint8_t nPool,bool check)
{
	if (!Status_Ok) return(false);

	ghMutexResources.lock();

	if ((!Status_Ok) || (nPool>=NbrePool))
	{
		ghMutexResources.unlock();
		return(false);
	}

	CriticalSection.lock();

	if (!Status_Ok)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}

	bool ok=true;

	if (check && (NbreUsers>0))
	{
		for (uint16_t i=0; i<NbreUsers; i++)
			ok=ok && !TabId[i].nPollTab[nPool];
	}
	if (ok) FreePool(nPool);

	CriticalSection.unlock();
	ghMutexResources.unlock();

	return(true);
}


bool ThreadPoolInterface::RemoveUserId(uint16_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1))
	{
		CriticalSection.unlock();
		return(false);
	}

	int8_t nPool=0;

	while(nPool<(int8_t)NbrePool)
	{
		if (TabId[index].nPollTab[nPool])
		{
			while (ThreadPoolRequested[nPool])
			{
				CriticalSection.unlock();
				WaitForSingleObject(ThreadPoolFree[nPool].get(),INFINITE);
				CriticalSection.lock();
				index=GetUserIdIndex(UserId);
				if ((!Status_Ok) || (index==-1))
				{
					CriticalSection.unlock();
					return(false);
				}
			}
		}
		nPool++;
	}

	if (index<NbreUsers-1)
	{
		for(uint16_t i=(uint16_t)index+1; i<NbreUsers; i++)
			TabId[i-1]=TabId[i];
	}
	NbreUsers--;
	TabId[NbreUsers].UserId=0;
	TabId[NbreUsers].nPool=-1;
	if (NbrePool>0)
	{
		for(uint8_t i=0; i<NbrePool; i++)
			TabId[NbreUsers].nPollTab[i]=false;
	}

	CriticalSection.unlock();

	return(true);
}


bool ThreadPoolInterface::DeAllocateUserThreads(uint16_t UserId,bool check)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	ghMutexResources.lock();

	if (!Status_Ok)
	{
		ghMutexResources.unlock();
		return(false);
	}

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1))
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}

	bool nPollTab[MAX_THREAD_POOL];

	memcpy(nPollTab,(void *)TabId[index].nPollTab,sizeof(nPollTab));

	if (index<NbreUsers-1)
	{
		for(uint16_t i=(uint16_t)index+1; i<NbreUsers; i++)
			TabId[i-1]=TabId[i];
	}
	NbreUsers--;
	TabId[NbreUsers].UserId=0;
	TabId[NbreUsers].nPool=-1;
	if (NbrePool>0)
	{
		for(uint8_t i=0; i<NbrePool; i++)
			TabId[NbreUsers].nPollTab[i]=false;
	}

	bool ok=true;

	int8_t nPool=0;

	if (check && (NbreUsers>0))
	{
		while(nPool<(int8_t)NbrePool)
		{
			if (nPollTab[nPool])
			{
				for (uint16_t i=0; i<NbreUsers; i++)
					ok=ok && !TabId[i].nPollTab[nPool];
			}
			nPool++;
		}
	}

	if (ok)
	{
		nPool=0;

		while(nPool<(int8_t)NbrePool)
		{
			if (nPollTab[nPool]) FreePool(nPool);
			nPool++;
		}
	}

	CriticalSection.unlock();
	ghMutexResources.unlock();

	return(true);
}


bool ThreadPoolInterface::DeAllocateAllThreads(bool check)
{
	if (!Status_Ok) return(false);

	ghMutexResources.lock();

	if (!Status_Ok)
	{
		ghMutexResources.unlock();
		return(false);
	}

	CriticalSection.lock();

	if (!Status_Ok)
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		return(false);
	}

	if (!(check && (NbreUsers>0))) FreePool();

	CriticalSection.unlock();
	ghMutexResources.unlock();

	return(true);
}


bool ThreadPoolInterface::RequestThreadPool(uint16_t UserId,uint8_t thread_number,Public_MT_Data_Thread *Data,int8_t nPool,bool Exclusive)
{
	if (!RequestThreadPool(UserId,thread_number,Data,nPool,Exclusive,false) || (nPool==-1)) return(false);
	else return(true);
}



bool ThreadPoolInterface::RequestThreadPool(uint16_t UserId,uint8_t thread_number,Public_MT_Data_Thread *Data,int8_t &nPool,bool Exclusive,bool AllowSeveral)
{
	if ((!Status_Ok) || Error_Occured || (UserId==0) || (nPool<-1) || (thread_number==0) || (Data==NULL))
	{
		nPool=-1;
		return(false);
	}

	ghMutexResources.lock();

	if ((!Status_Ok) || Error_Occured  || (nPool>=(int8_t)NbrePool))
	{
		ghMutexResources.unlock();
		nPool=-1;
		return(false);
	}

	CriticalSection.lock();

	int16_t userindex=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (userindex==-1))
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		nPool=-1;
		return(false);
	}
	
	int8_t nP=TabId[userindex].nPool;

	if ((nP<-1) || (nP>=(int8_t)NbrePool))
	{
		CriticalSection.unlock();
		ghMutexResources.unlock();
		nPool=-1;
		return(false);
	}

	if (nP>=0)
	{
		if ((!ThreadPoolRequested[nP]) || (ThreadPoolUserId[nP]!=UserId)) 
		{
			CriticalSection.unlock();
			ghMutexResources.unlock();
			nPool=-1;
			return(false);
		}

		if ((!AllowSeveral) && (thread_number==ptrPool[nP]->GetCurrentThreadUsed()))
		{
			CriticalSection.unlock();
			ghMutexResources.unlock();
			nPool=nP;
			return(true);
		}

		if ((nPool>=0) && (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated()))
		{
			CriticalSection.unlock();
			ghMutexResources.unlock();
			nPool=-1;
			return(false);
		}
	}

	bool dealocate_curent=false;

	if (nPool==-1)
	{
		bool check_ok=false;

		for(uint8_t i=0; i<NbrePool; i++)
			check_ok=check_ok || (thread_number<=ptrPool[i]->GetCurrentThreadAllocated());
		if (!check_ok)
		{
			CriticalSection.unlock();
			ghMutexResources.unlock();
			nPool=-1;
			return(false);
		}

		if ((!AllowSeveral) && (nP>=0)) dealocate_curent=true;
	}
	else
	{
		if (nP==-1)
		{
			if (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated())
			{
				CriticalSection.unlock();
				ghMutexResources.unlock();
				nPool=-1;
				return(false);
			}
		}
		else dealocate_curent=!AllowSeveral;
	}
	
	if (dealocate_curent)
	{
		CriticalSection.unlock();
		if (!ReleaseThreadPool(UserId,true))
		{
			ghMutexResources.unlock();
			nPool=-1;
			return(false);
		}
		CriticalSection.lock();
		userindex=GetUserIdIndex(UserId);
		if ((!Status_Ok) || Error_Occured || (userindex==-1))
		{
			CriticalSection.unlock();
			ghMutexResources.unlock();
			nPool=-1;
			return(false);
		}
		ThreadPoolUserId[nP]=0;
		TabId[userindex].nPollTab[nP]=false;
		TabId[userindex].nPool=-1;
	}

	while (ExclusiveMode)
	{
		CriticalSection.unlock();
		WaitForSingleObject(EndExclusive.get(),INFINITE);
		CriticalSection.lock();
		userindex=GetUserIdIndex(UserId);
		if ((!Status_Ok) || Error_Occured || (userindex==-1))
		{
			CriticalSection.unlock();
			ghMutexResources.unlock();
			nPool=-1;
			return(false);
		}
	}

	if (nPool==-1)
	{
		if ((NbrePool>1) && (!Exclusive))
		{
			auto findFreePool = [&]() -> int8_t {
				for (uint8_t i = 0; i < NbrePool; i++)
				{
					if (thread_number <= ptrPool[i]->GetCurrentThreadAllocated() && !ThreadPoolRequested[i])
						return (int8_t)i;
				}
				return -1;
			};

			nPool = findFreePool();
			while (nPool == -1)
			{
				std::unique_lock<std::mutex> stateLock(CriticalSection, std::adopt_lock);
				PoolStateChanged.wait(stateLock, [&]() { return findFreePool() != -1; });
				stateLock.release();
				userindex=GetUserIdIndex(UserId);
				if ((!Status_Ok) || Error_Occured || (userindex==-1))
				{
					CriticalSection.unlock();
					ghMutexResources.unlock();
					nPool=-1;
					return(false);
				}
				nPool = findFreePool();
			}
		}
		else
		{
			int8_t i=0;
			while ((i<NbrePool) && (thread_number>ptrPool[i]->GetCurrentThreadAllocated())) i++;
			nPool=i;
		}
	}

	if ((!Exclusive) || (NbrePool==1))
	{
		while (ThreadPoolRequested[nPool])
		{
			CriticalSection.unlock();
			WaitForSingleObject(ThreadPoolFree[nPool].get(),INFINITE);
			CriticalSection.lock();
			userindex=GetUserIdIndex(UserId);
			if ((!Status_Ok) || Error_Occured || (userindex==-1))
			{
				CriticalSection.unlock();
				ghMutexResources.unlock();
				nPool=-1;
				return(false);
			}
		}
	}
	else
	{
		bool PoolFree=true;
		
		for(uint8_t i=0; i<NbrePool; i++)
			PoolFree=PoolFree && (!ThreadPoolRequested[i]);		

		while (!PoolFree)
		{
			HANDLE handles[MAX_THREAD_POOL];
			for (uint8_t i = 0; i < NbrePool; i++) {
				handles[i] = ThreadPoolFree[i].get();
			}
			
			CriticalSection.unlock();
			WaitForMultipleObjects(NbrePool, handles, TRUE, INFINITE);
			CriticalSection.lock();
			userindex=GetUserIdIndex(UserId);
			if ((!Status_Ok) || Error_Occured || (userindex==-1))
			{
				CriticalSection.unlock();
				ghMutexResources.unlock();
				nPool=-1;
				return(false);
			}
			PoolFree=true;
			for(uint8_t i=0; i<NbrePool; i++)
				PoolFree=PoolFree && (!ThreadPoolRequested[i]);
		}
	}
	
	bool out=ptrPool[nPool]->RequestThreadPool(thread_number,Data);
	
	if (out)
	{
		ThreadPoolRequested[nPool]=true;
		ResetEvent(ThreadPoolFree[nPool].get());
		if (Exclusive)
		{
			ExclusiveMode=true;
			ResetEvent(EndExclusive.get());
		}
		if (TabId[userindex].nPool==-1) TabId[userindex].nPool=nPool;
		ThreadPoolUserId[nPool]=UserId;
		TabId[userindex].nPollTab[nPool]=true;
	}
	else nPool=-1;
	
	CriticalSection.unlock();
	ghMutexResources.unlock();

	return(out);	
}


bool ThreadPoolInterface::ReleaseThreadPool(uint16_t UserId,bool sleep)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1))
	{
		CriticalSection.unlock();
		return(false);
	}

	int8_t nPool=TabId[index].nPool;

	if ((nPool<-1) || (nPool>=(int8_t)NbrePool))
	{
		CriticalSection.unlock();
		return(false);
	}

	if (nPool==-1)
	{
		CriticalSection.unlock();
		return(true);
	}

	return(ReleaseThreadPoolCore(UserId,index,sleep,nPool));
}


bool ThreadPoolInterface::ReleaseThreadPool(uint16_t UserId,bool sleep,int8_t nPool)
{
	if ((!Status_Ok) || (UserId==0) || (nPool<0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1) || (nPool>=(int8_t)NbrePool))
	{
		CriticalSection.unlock();
		return(false);
	}

	if (ThreadPoolUserId[nPool]==0)
	{
		CriticalSection.unlock();
		return(true);
	}

	if (ThreadPoolUserId[nPool]!=UserId)
	{
		CriticalSection.unlock();
		return(false);
	}

	return(ReleaseThreadPoolCore(UserId,index,sleep,nPool));
}


bool ThreadPoolInterface::ReleaseThreadPoolCore(uint16_t UserId,int16_t index,bool sleep,int8_t nPool)
{
	bool out=true;

	ThreadPoolReleased[nPool]=true;

	if (ThreadPoolRequested[nPool])
	{
		while (JobsRunning[nPool])
		{
			HANDLE h=JobsEnded[nPool].get();
			
			CriticalSection.unlock();
			WaitForSingleObject(h,INFINITE);
			CriticalSection.lock();
			index=GetUserIdIndex(UserId);
			if ((!Status_Ok) || (index==-1))
			{
				ThreadPoolReleased[nPool]=false;				
				CriticalSection.unlock();
				return(false);
			}
		}
		ThreadPoolRequested[nPool]=false;
		out=ptrPool[nPool]->ReleaseThreadPool(sleep);
		SetEvent(ThreadPoolFree[nPool].get());
		PoolStateChanged.notify_all();
	}

	if (ExclusiveMode)
	{
		ExclusiveMode=false;
		SetEvent(EndExclusive.get());
	}

	ThreadPoolReleased[nPool]=false;
	ThreadPoolUserId[nPool]=0;
	if (TabId[index].nPool==nPool) TabId[index].nPool=-1;
	TabId[index].nPollTab[nPool]=false;

	CriticalSection.unlock();

	return(out);
}


bool ThreadPoolInterface::StartThreads(uint16_t UserId)
{
	if ((!Status_Ok) || Error_Occured || (UserId==0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (index==-1))
	{
		CriticalSection.unlock();
		return(false);
	}

	int8_t nPool=TabId[index].nPool;

	if ((nPool<0) || (nPool>=(int8_t)NbrePool))
	{
		CriticalSection.unlock();
		return(false);
	}

	return(StartThreadsCore(nPool));
}



bool ThreadPoolInterface::StartThreads(uint16_t UserId,int8_t nPool)
{
	if ((!Status_Ok) || Error_Occured || (UserId==0) || (nPool<0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (index==-1) || (nPool>=(int8_t)NbrePool) || (ThreadPoolUserId[nPool]!=UserId))
	{
		CriticalSection.unlock();
		return(false);
	}

	return(StartThreadsCore(nPool));
}


bool ThreadPoolInterface::StartThreadsCore(int8_t nPool)
{
	if ((!ThreadPoolRequested[nPool]) || ThreadPoolReleased[nPool] || ThreadWaitEnd[nPool] || ThreadPoolWaitFree[nPool])
	{
		CriticalSection.unlock();
		return(false);
	}

	if (JobsRunning[nPool])
	{
		CriticalSection.unlock();
		return(true);
	}

	bool out=ptrPool[nPool]->StartThreads();

	if (out)
	{
		JobsRunning[nPool]=true;
		ResetEvent(JobsEnded[nPool].get());
	}

	CriticalSection.unlock();

	return(out);	
}


bool ThreadPoolInterface::WaitThreadsEnd(uint16_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if  ((!Status_Ok) || (index==-1))
	{
		CriticalSection.unlock();
		return(false);
	}

	int8_t nPool=TabId[index].nPool;

	if ((nPool<0) || (nPool>=(int8_t)NbrePool))
	{
		CriticalSection.unlock();
		return(false);
	}

	return(WaitThreadsEndCore(UserId,nPool));
}


bool ThreadPoolInterface::WaitThreadsEnd(uint16_t UserId,int8_t nPool)
{
	if ((!Status_Ok) || (UserId==0) || (nPool<0)) return(false);

	CriticalSection.lock();

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1) || (nPool>=(int8_t)NbrePool) || (ThreadPoolUserId[nPool]!=UserId))
	{
		CriticalSection.unlock();
		return(false);
	}

	return(WaitThreadsEndCore(UserId,nPool));
}


bool ThreadPoolInterface::WaitThreadsEndCore(uint16_t UserId,int8_t nPool)
{
	if ((!ThreadPoolRequested[nPool]) || ThreadWaitEnd[nPool])
	{
		CriticalSection.unlock();
		return(false);
	}

	if (!JobsRunning[nPool])
	{
		CriticalSection.unlock();
		return(true);
	}
	
	ThreadWaitEnd[nPool]=true;

	ThreadPool *ptr=ptrPool[nPool];
	
	CriticalSection.unlock();
	bool out=ptr->WaitThreadsEnd();
	CriticalSection.lock();

	ThreadWaitEnd[nPool]=false;

	int8_t nPool2=-1;

	int16_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1))
	{
		CriticalSection.unlock();
		return(false);
	}

	if (out)
	{
		JobsRunning[nPool]=false;
		SetEvent(JobsEnded[nPool].get());
	}
	
	CriticalSection.unlock();

	return(out);
}


bool ThreadPoolInterface::GetThreadPoolStatus(uint16_t UserId,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (NbreUsers==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetThreadPoolStatus());
			else return(false);
		}
		else
		{
			int16_t i=GetUserIdIndex(UserId);

			if (i==-1)
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetThreadPoolStatus());
				else return(false);
			}
			if ((TabId[i].nPool>=0) && (TabId[i].nPool<(int8_t)NbrePool))
				return(ptrPool[TabId[i].nPool]->GetThreadPoolStatus());
			else return(false);		
		}
	}
	else return(false);
}


uint8_t ThreadPoolInterface::GetCurrentThreadAllocated(uint16_t UserId,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (NbreUsers==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadAllocated());
			else return(0);
		}
		else
		{
			int16_t i=GetUserIdIndex(UserId);

			if (i==-1)
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadAllocated());
				else return(0);
			}
			if ((TabId[i].nPool>=0) && (TabId[i].nPool<(int8_t)NbrePool))
				return(ptrPool[TabId[i].nPool]->GetCurrentThreadAllocated());
			else return(0);		
		}
	}
	else return(0);
}


uint8_t ThreadPoolInterface::GetCurrentThreadUsed(uint16_t UserId,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (NbreUsers==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadUsed());
			else return(0);
		}
		else
		{
			int16_t i=GetUserIdIndex(UserId);

			if (i==-1)
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadUsed());
				else return(0);
			}
			if ((TabId[i].nPool>=0) && (TabId[i].nPool<(int8_t)NbrePool))
				return(ptrPool[TabId[i].nPool]->GetCurrentThreadUsed());
			else return(0);		
		}
	}
	else return(0);
}


uint8_t ThreadPoolInterface::GetLogicalCPUNumber(void)
{
	if (Status_Ok && (NbrePool>0)) return(ptrPool[0]->GetLogicalCPUNumber());
	else return(0);
}


uint8_t ThreadPoolInterface::GetPhysicalCoreNumber(void)
{
	if (Status_Ok && (NbrePool>0)) return(ptrPool[0]->GetPhysicalCoreNumber());
	else return(0);
}
