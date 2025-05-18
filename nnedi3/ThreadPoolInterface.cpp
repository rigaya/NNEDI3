/*
 *  ThreadpoolInterface
 *
 *  Allow to use the threadpool, kind of API.
 *  Copyright (C) 2017 JPSDR
 *	
 *  ThreadpoolInterface is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *   
 *  ThreadpoolInterface is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *   
 *  You should have received a copy of the GNU General Public License
 *  along with GNU Make; see the file COPYING.  If not, write to
 *  the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA. 
 *
 */

#include "./ThreadPoolInterface.h"
#include "./ThreadPool.h"

#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}
#define mydelete(ptr) if (ptr!=NULL) { delete ptr; ptr=NULL;}

// OSごとの違いを吸収するためのマクロ
#if !defined(_WIN32) && !defined(_WIN64)
#define WAIT_FAILED 0xFFFFFFFF
#define WAIT_TIMEOUT 258L
#define WAIT_ABANDONED_0 0x00000080L
#define WAIT_OBJECT_0 0
#endif

static ThreadPool *ptrPool[MAX_THREAD_POOL];

UserData::UserData(void)
{
	UserId=0;
	AllowSeveral=false;
	AllowWaiting=true;
	AllowTimeOut=false;
	AllowRetryMax=false;
	TimeOut=100;
	RetryMax=2;
	NbrePool=0;
	for (uint8_t i=0; i<MAX_THREAD_POOL; i++)
		UsedPool[i]=-1;
}

UserData::~UserData(void) {};

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
			JobsEnded[NbrePoolEvent] = CreateEventUnique(nullptr, 1, 1);
			ThreadPoolFree[NbrePoolEvent] = CreateEventUnique(nullptr, 1, 1);
			ok = ok && (JobsEnded[NbrePoolEvent].get() != NULL) && (ThreadPoolFree[NbrePoolEvent].get() != NULL);
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

			const std::vector<UserData>::size_type NbreUsers=TabId.size();

			if (NbreUsers>0)
			{
				for(std::vector<UserData>::size_type i=0; i<NbreUsers; i++)
				{
					int8_t j=0;

					while ((j<MAX_THREAD_POOL) && (TabId[i].UsedPool[j]!=nPool)) j++;
					if (j<MAX_THREAD_POOL)
					{
						TabId[i].UsedPool[j]=-1;
						TabId[i].NbrePool--;
					}
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

	const std::vector<UserData>::size_type NbreUsers=TabId.size();

	if (NbreUsers>0)
	{
		for(std::vector<UserData>::size_type i=0; i<NbreUsers; i++)
		{
			TabId[i].NbrePool=0;
			for(uint8_t j=0; j<MAX_THREAD_POOL; j++)
				TabId[i].UsedPool[j]=-1;
		}
	}
}


ThreadPoolInterface::ThreadPoolInterface(void):Status_Ok(false),Error_Occured(false),
	NbrePool(0),NbrePoolEvent(0),
	EndExclusive(CreateEventUnique(nullptr, 1, 1))
{
	Status_Ok = true;
	if (!EndExclusive) {
		Status_Ok = false;
	}

	// JobsEnded, ThreadPoolFreeの配列と同等のサイズ設定
	for (uint8_t i=0; i<MAX_THREAD_POOL; i++) {
		JobsEnded.push_back(unique_event(nullptr, CloseEvent));
		ThreadPoolFree.push_back(unique_event(nullptr, CloseEvent));
	}

	ExclusiveMode=false;
	TabId.clear();

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
}


ThreadPoolInterface::~ThreadPoolInterface(void)
{
	FreeData();
	TabId.clear();
}


uint8_t ThreadPoolInterface::GetThreadNumber(uint8_t thread_number,bool logical)
{
	if ((!Status_Ok) || (NbrePool==0)) return(0);
	else return(ptrPool[0]->GetThreadNumber(thread_number,logical));
}


bool ThreadPoolInterface::CreatePool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured || (num==0) || (num>MAX_THREAD_POOL)) return(false);
	
	GetMutex();

	if ((!Status_Ok) || Error_Occured) 
	{
		FreeMutex();
		return(false);
	}
	
	EnterCS();
	if ((!Status_Ok) || Error_Occured)
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}

	if (num<=NbrePool)
	{
		LeaveCS();
		FreeMutex();
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
	
	LeaveCS();
	FreeMutex();
	
	return(Status_Ok);
}


int16_t ThreadPoolInterface::AddPool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured || (num==0)) return(-1);
	
	GetMutex();

	if ((!Status_Ok) || Error_Occured || ((NbrePool+num)>=MAX_THREAD_POOL)) 
	{
		FreeMutex();
		return(-1);
	}
	
	EnterCS();
	if ((!Status_Ok) || Error_Occured)
	{
		LeaveCS();
		FreeMutex();
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
	
	LeaveCS();
	FreeMutex();
	
	return(CurrentPool);
}


bool ThreadPoolInterface::DeletePool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured || (num==0)) return(false);
	
	GetMutex();

	if ((!Status_Ok) || Error_Occured || (num>NbrePool))
	{
		FreeMutex();
		return(false);
	}
	
	EnterCS();
	if ((!Status_Ok) || Error_Occured)
	{
		LeaveCS();
		FreeMutex();
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
				LeaveCS();
				WaitForSingleObject(ThreadPoolFree[CurrentNum].get(), INFINITE);
				EnterCS();
			}
			ptrPool[CurrentNum]->DeAllocateThreads();

			const std::vector<UserData>::size_type NbreUsers=TabId.size();

			if (NbreUsers>0)
			{
				for (std::vector<UserData>::size_type j=0; j<NbreUsers; j++)
				{
					int8_t k=0;

					while ((k<MAX_THREAD_POOL) && (TabId[j].UsedPool[k]!=(int8_t)CurrentNum)) k++;
					if (k<MAX_THREAD_POOL)
					{
						TabId[j].UsedPool[k]=-1;
						TabId[j].NbrePool--;
					}
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
	
	LeaveCS();
	FreeMutex();
	
	return(true);
}


bool ThreadPoolInterface::RemovePool(uint8_t num)
{
	if ((!Status_Ok) || Error_Occured) return(false);
	
	GetMutex();

	if ((!Status_Ok) || Error_Occured || (num>=NbrePool))
	{
		FreeMutex();
		return(false);
	}
	
	EnterCS();
	if ((!Status_Ok) || Error_Occured)
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}
	
	if (ptrPool[num]!=NULL)
	{
		ThreadPoolWaitFree[num]=true;
		while(ThreadPoolRequested[num])
		{
			LeaveCS();
			WaitForSingleObject(ThreadPoolFree[num].get(), INFINITE);
			EnterCS();
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
	
	const std::vector<UserData>::size_type NbreUsers=TabId.size();

	if (NbreUsers>0)
	{
		for (std::vector<UserData>::size_type i=0; i<NbreUsers; i++)
		{
			for(int8_t j=0; j<MAX_THREAD_POOL; j++)
			{
				if (TabId[i].UsedPool[j]==(int8_t)num)
				{
					TabId[i].UsedPool[j]=-1;
					TabId[i].NbrePool--;
				}
				else
				{
					if (TabId[i].UsedPool[j]>(int8_t)num) TabId[i].UsedPool[j]--;
				}
			}
		}
	}
	
	NbrePool--;
	
	LeaveCS();
	FreeMutex();
	
	return(true);
}


bool ThreadPoolInterface::AllocateThreads(uint8_t thread_number,uint8_t offset_core,uint8_t offset_ht,
	bool UseMaxPhysCore,bool SetAffinity,bool sleep,ThreadLevelName priority,int8_t nPool)
{
	if ((!Status_Ok) || Error_Occured || (thread_number==0) || (nPool<-1)) return(false);

	GetMutex();

	if ((!Status_Ok) || Error_Occured  || (NbrePool==0) || (nPool>=(int8_t)NbrePool))
	{
		FreeMutex();
		return(false);
	}

	EnterCS();
	if ((!Status_Ok) || Error_Occured)
	{
		LeaveCS();
		FreeMutex();
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
					LeaveCS();
					WaitForSingleObject(JobsEnded[i].get(),INFINITE);
					EnterCS();
					if ((!Status_Ok) || Error_Occured)
					{
						LeaveCS();
						FreeMutex();
						return(false);
					}
				}
				bool ok=ptrPool[i]->AllocateThreads(thread_number,offset_core,offset_ht,UseMaxPhysCore,
					SetAffinity,sleep,priority);
				if (!ok)
				{
					Error_Occured=true;
					FreePool();
					Status_Ok=false;
					FreeData();
					LeaveCS();
					FreeMutex();
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
				LeaveCS();
				WaitForSingleObject(JobsEnded[nPool].get(),INFINITE);
				EnterCS();
				if ((!Status_Ok) || Error_Occured)
				{
					LeaveCS();
					FreeMutex();
					return(false);
				}
			}
			bool ok=ptrPool[nPool]->AllocateThreads(thread_number,offset_core,offset_ht,UseMaxPhysCore,
				SetAffinity,sleep,priority);
			if (!ok)
			{
				Error_Occured=true;
				FreePool();
				Status_Ok=false;
				FreeData();
				LeaveCS();
				FreeMutex();
				return(false);
			}
		}
	}

	LeaveCS();
	FreeMutex();

	return(true);
}


inline int32_t ThreadPoolInterface::GetUserIdIndex(uint32_t UserId)
{
	const std::vector<UserData>::size_type NbreUsers=TabId.size();

	if ((UserId==0) || (NbreUsers==0)) return(-1);

	std::vector<UserData>::size_type i=0;

	while ((NbreUsers>i) && (TabId[i].UserId!=UserId)) i++;

	if (i==NbreUsers) return(-1);
	else return((int32_t)i);
}


bool ThreadPoolInterface::GetUserId(uint32_t &UserId)
{
	if ((!Status_Ok) || Error_Occured) return(false);

	EnterCS();
	if ((!Status_Ok) || Error_Occured)
	{
		LeaveCS();
		return(false);
	}

	const std::vector<UserData>::size_type NbreUsers=TabId.size();

	bool ret_status=true;

	if (UserId==0)
	{
		uint32_t user=1;

		if (NbreUsers>0)
		{
			bool found=false;

			while (!found && (user<=NbreUsers))
			{
				std::vector<UserData>::size_type i=0;
				bool search=true;

				while (search && (i<NbreUsers))
					search=search && (TabId[i++].UserId!=user);
				found=search;
				if (!found) user++;
			}
		}

		UserData userD;

		userD.UserId=user;
		TabId.push_back(userD);

		ret_status=TabId.size()>NbreUsers;
		if (ret_status) UserId=user;
	}
	else ret_status=(GetUserIdIndex(UserId)!=-1);

	LeaveCS();

	return(ret_status);
}


bool ThreadPoolInterface::ChangeThreadsAffinity(uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,bool SetAffinity,int8_t nPool)
{
	if ((!Status_Ok) || Error_Occured || (nPool<-1)) return(false);

	GetMutex();

	if ((!Status_Ok) || Error_Occured || (nPool>=(int8_t)NbrePool))
	{
		FreeMutex();
		return(false);
	}

	EnterCS();
	if ((!Status_Ok) || Error_Occured )
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}

	if (nPool==-1)
	{
		for(uint8_t i=0; i<NbrePool; i++)
		{
			if (ptrPool[i]->GetCurrentThreadAllocated()>0)
			{
				bool ok=ptrPool[i]->ChangeThreadsAffinity(offset_core,offset_ht,UseMaxPhysCore,SetAffinity);
				if (!ok)
				{
					Error_Occured=true;
					FreePool();
					Status_Ok=false;
					FreeData();
					LeaveCS();
					FreeMutex();
					return(false);
				}
			}
		}
	}
	else
	{
		if (ptrPool[nPool]->GetCurrentThreadAllocated()>0)
		{
			bool ok=ptrPool[nPool]->ChangeThreadsAffinity(offset_core,offset_ht,UseMaxPhysCore,SetAffinity);
			if (!ok)
			{
				Error_Occured=true;
				FreePool();
				Status_Ok=false;
				FreeData();
				LeaveCS();
				FreeMutex();
				return(false);
			}
		}
	}

	LeaveCS();
	FreeMutex();

	return(true);
}


bool ThreadPoolInterface::ChangeThreadsLevel(ThreadLevelName priority,int8_t nPool)
{
	if ((!Status_Ok) || Error_Occured || (nPool<-1)) return(false);

	GetMutex();

	if ((!Status_Ok) || Error_Occured || (nPool>=(int8_t)NbrePool))
	{
		FreeMutex();
		return(false);
	}

	EnterCS();
	if ((!Status_Ok) || Error_Occured )
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}

	if (nPool==-1)
	{
		for(uint8_t i=0; i<NbrePool; i++)
		{
			if (ptrPool[i]->GetCurrentThreadAllocated()>0)
			{
				bool ok=ptrPool[i]->ChangeThreadsLevel(priority);
				if (!ok)
				{
					Error_Occured=true;
					FreePool();
					Status_Ok=false;
					FreeData();
					LeaveCS();
					FreeMutex();
					return(false);
				}
			}
		}
	}
	else
	{
		if (ptrPool[nPool]->GetCurrentThreadAllocated()>0)
		{
			bool ok=ptrPool[nPool]->ChangeThreadsLevel(priority);
			if (!ok)
			{
				Error_Occured=true;
				FreePool();
				Status_Ok=false;
				FreeData();
				LeaveCS();
				FreeMutex();
				return(false);
			}
		}
	}

	LeaveCS();
	FreeMutex();

	return(true);
}


bool ThreadPoolInterface::DeAllocatePoolThreads(uint8_t nPool,bool check)
{
	if ((!Status_Ok) || (nPool<0)) return(false);

	GetMutex();

	if ((!Status_Ok) || (nPool>=NbrePool))
	{
		FreeMutex();
		return(false);
	}

	EnterCS();

	if (!Status_Ok)
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}

	bool ok=true;

	const std::vector<UserData>::size_type NbreUsers=TabId.size();

	if (check && (NbreUsers>0))
	{
		for (std::vector<UserData>::size_type i=0; i<NbreUsers; i++)
			for(int8_t j=0; j<MAX_THREAD_POOL; j++)
				ok=ok && (TabId[i].UsedPool[j]!=nPool);
	}
	if (ok) FreePool(nPool);

	LeaveCS();
	FreeMutex();

	return(true);
}


bool ThreadPoolInterface::RemoveUserId(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1))
	{
		LeaveCS();
		return(false);
	}

	for(int8_t j=0; j<MAX_THREAD_POOL; j++)
	{
		int8_t nPool=TabId[index].UsedPool[j];

		if ((nPool>=0) && (nPool<(int8_t)NbrePool) && (ThreadPoolUserId[nPool]==UserId))
		{
			while (ThreadPoolRequested[nPool])
			{
				LeaveCS();
				WaitForSingleObject(ThreadPoolFree[nPool].get(),INFINITE);
				EnterCS();
				index=GetUserIdIndex(UserId);
				if ((!Status_Ok) || (index==-1))
				{
					LeaveCS();
					return(false);
				}
			}
		}
	}

	std::vector<UserData>::iterator it=TabId.begin()+(std::vector<UserData>::size_type)index;

	TabId.erase(it);

	LeaveCS();

	return(true);
}


bool ThreadPoolInterface::DeAllocateUserThreads(uint32_t UserId,bool check)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	GetMutex();

	if (!Status_Ok)
	{
		FreeMutex();
		return(false);
	}

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1))
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}

	int8_t UsedPool[MAX_THREAD_POOL];

	memcpy(UsedPool,(void *)TabId[index].UsedPool,sizeof(UsedPool));

	std::vector<UserData>::iterator it=TabId.begin()+(std::vector<UserData>::size_type)index;

	TabId.erase(it);

	const std::vector<UserData>::size_type NbreUsers=TabId.size();

	bool ok=true;

	if (check && (NbreUsers>0))
	{
		for(int8_t i=0; i<MAX_THREAD_POOL; i++)
		{
			int8_t nPool=UsedPool[i];

			if ((nPool>=0) && (nPool<(int8_t)NbrePool) && (ThreadPoolUserId[nPool]==UserId))
			{
				for (std::vector<UserData>::size_type j=0; j<NbreUsers; j++)
					for(int8_t k=0; k<MAX_THREAD_POOL; k++)
						ok=ok && (TabId[j].UsedPool[k]!=nPool);
			}
		}
	}

	if (ok)
	{
		for(int8_t i=0; i<MAX_THREAD_POOL; i++)
		{
			int8_t nPool=UsedPool[i];

			if ((nPool>=0) && (nPool<(int8_t)NbrePool) && (ThreadPoolUserId[nPool]==UserId)) FreePool(nPool);
		}
	}

	LeaveCS();
	FreeMutex();

	return(true);
}


bool ThreadPoolInterface::DeAllocateAllThreads(bool check)
{
	if (!Status_Ok) return(false);

	GetMutex();

	if (!Status_Ok)
	{
		FreeMutex();
		return(false);
	}

	EnterCS();

	if (!Status_Ok)
	{
		LeaveCS();
		FreeMutex();
		return(false);
	}

	if (!(check && (TabId.size()>0))) FreePool();

	LeaveCS();
	FreeMutex();

	return(true);
}


bool ThreadPoolInterface::RequestThreadPool(uint32_t UserId,uint8_t thread_number,Public_MT_Data_Thread *Data,
	ThreadLevelName priority,int8_t nPool,bool Exclusive)
{
	int8_t idx;

	if (!RequestThreadPool(UserId,idx,thread_number,Data,priority,nPool,Exclusive) || (nPool==-1) || (idx==-1)) return(false);
	else return(true);
}


bool ThreadPoolInterface::RequestThreadPool(uint32_t UserId,int8_t &idxPool,uint8_t thread_number,Public_MT_Data_Thread *Data)
{
	int8_t nPool=-1;

	if (!RequestThreadPool(UserId,idxPool,thread_number,Data,NoneThreadLevel,nPool,false) || (nPool==-1)) return(false);
	else return(true);
}


bool ThreadPoolInterface::RequestThreadPool(uint32_t UserId,int8_t &idxPool,uint8_t thread_number,Public_MT_Data_Thread *Data,
	ThreadLevelName priority)
{
	int8_t nPool=-1;

	if (!RequestThreadPool(UserId,idxPool,thread_number,Data,priority,nPool,false) || (nPool==-1)) return(false);
	else return(true);
}


bool ThreadPoolInterface::RequestThreadPool(uint32_t UserId,int8_t &idxPool,uint8_t thread_number,Public_MT_Data_Thread *Data,
	ThreadLevelName priority,int8_t &nPool,bool Exclusive)
{
	bool AllowSeveral,AllowWaiting;

	idxPool=-1;

	if ((!Status_Ok) || Error_Occured || (UserId==0) || (nPool<-1) || (thread_number==0) || (Data==NULL))
	{
		nPool=-1;
		return(false);
	}

	GetMutex();

	if ((!Status_Ok) || Error_Occured  || (nPool>=(int8_t)NbrePool))
	{
		FreeMutex();
		nPool=-1;
		return(false);
	}

	EnterCS();

	int32_t userindex=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (userindex==-1))
	{
		LeaveCS();
		FreeMutex();
		nPool=-1;
		return(false);
	}

	AllowSeveral=TabId[userindex].AllowSeveral;
	AllowWaiting=TabId[userindex].AllowWaiting;

	if (TabId[userindex].NbrePool>0)
	{
		for (int8_t i=0; i<MAX_THREAD_POOL; i++)
		{
			int8_t nP=TabId[userindex].UsedPool[i];

			if ((nP<-1) || (nP>=(int8_t)NbrePool))
			{
				LeaveCS();
				FreeMutex();
				nPool=-1;
				return(false);
			}

			if (nP>=0) 
			{
				if ((!ThreadPoolRequested[nP]) || (ThreadPoolUserId[nP]!=UserId)) 
				{
					LeaveCS();
					FreeMutex();
					nPool=-1;
					return(false);
				}

				if ((!AllowSeveral) && (thread_number==ptrPool[nP]->GetCurrentThreadUsed()))
				{
					LeaveCS();
					FreeMutex();
					nPool=nP;
					idxPool=i;
					return(true);
				}
			}
		}

		if ((nPool>=0) && (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated()))
		{
			LeaveCS();
			FreeMutex();
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
			LeaveCS();
			FreeMutex();
			nPool=-1;
			return(false);
		}

		if ((!AllowSeveral) && (TabId[userindex].NbrePool>0)) dealocate_curent=true;
	}
	else
	{
		if (TabId[userindex].NbrePool==0)
		{
			if (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated())
			{
				LeaveCS();
				FreeMutex();
				nPool=-1;
				return(false);
			}
		}
		else dealocate_curent=!AllowSeveral;
	}
	
	if (dealocate_curent)
	{
		LeaveCS();
		if (!ReleaseThreadPool(UserId,true))
		{
			FreeMutex();
			nPool=-1;
			return(false);
		}
		EnterCS();
		userindex=GetUserIdIndex(UserId);
		if ((!Status_Ok) || Error_Occured || (userindex==-1))
		{
			LeaveCS();
			FreeMutex();
			nPool=-1;
			return(false);
		}
	}

	if (!AllowWaiting)
	{
		uint8_t NbUsed=0;

		for (uint8_t i=0; i<NbrePool; i++)
			if (ThreadPoolRequested[i]) NbUsed++;

		if (NbUsed==NbrePool)
		{
			LeaveCS();
			FreeMutex();
			nPool=-1;
			return(false);
		}
	}

	const DWORD TimeOutWait=(TabId[userindex].AllowTimeOut)?TabId[userindex].TimeOut:INFINITE;
	const bool EnableCount=TabId[userindex].AllowRetryMax;
	const uint8_t NbreMaxRetry=TabId[userindex].RetryMax;
	DWORD resultWait;
	uint8_t CptRetry=0;

	while (ExclusiveMode && (CptRetry<NbreMaxRetry))
	{
		LeaveCS();
		resultWait=WaitForSingleObject(EndExclusive.get(),TimeOutWait);
		EnterCS();
		userindex=GetUserIdIndex(UserId);
		if ((!Status_Ok) || Error_Occured || (userindex==-1)
			|| (resultWait==WAIT_TIMEOUT) || (resultWait==WAIT_FAILED))
		{
			LeaveCS();
			FreeMutex();
			nPool=-1;
			return(false);
		}
		if (EnableCount) CptRetry++;
	}
	if (ExclusiveMode)
	{
		LeaveCS();
		FreeMutex();
		nPool=-1;
		return(false);
	}

	CptRetry=0;
	if (nPool==-1)
	{
		if ((NbrePool>1) && (!Exclusive))
		{
			HANDLE TabTemp[MAX_THREAD_POOL];
			uint8_t TabNbre[MAX_THREAD_POOL];

			uint8_t Nbre=0;
			for(uint8_t i=0; i<NbrePool; i++)
			{
				if (thread_number<=ptrPool[i]->GetCurrentThreadAllocated())
				{
					TabTemp[Nbre]=ThreadPoolFree[i].get();
					TabNbre[Nbre++]=i;
					if (!ThreadPoolRequested[i]) nPool=i;
				}
			}

			bool PoolFree=(nPool==-1)?false:true;
		
			while ((!PoolFree) && (CptRetry<NbreMaxRetry))
			{	
				LeaveCS();
				resultWait=WaitForMultipleObjects(Nbre,TabTemp,FALSE,TimeOutWait);
				EnterCS();
				userindex=GetUserIdIndex(UserId);
				if ((!Status_Ok) || Error_Occured || (userindex==-1)
					|| (resultWait==WAIT_TIMEOUT) || (resultWait==WAIT_FAILED))
				{
					LeaveCS();
					FreeMutex();
					nPool=-1;
					return(false);
				}
				nPool=(int8_t)TabNbre[(resultWait-WAIT_OBJECT_0)];
				PoolFree=!ThreadPoolRequested[nPool];
				if (EnableCount) CptRetry++;
			}
			if (!PoolFree)
			{
				LeaveCS();
				FreeMutex();
				nPool=-1;
				return(false);
			}
		}
		else
		{
			int8_t i=0;
			while ((i<NbrePool) && (thread_number>ptrPool[i]->GetCurrentThreadAllocated())) i++;
			nPool=i;
		}
	}

	CptRetry=0;
	if ((!Exclusive) || (NbrePool==1))
	{
		while (ThreadPoolRequested[nPool] && (CptRetry<NbreMaxRetry))
		{
			LeaveCS();
			resultWait=WaitForSingleObject(ThreadPoolFree[nPool].get(),TimeOutWait);
			EnterCS();
			userindex=GetUserIdIndex(UserId);
			if ((!Status_Ok) || Error_Occured || (userindex==-1)
				|| (resultWait==WAIT_TIMEOUT) || (resultWait==WAIT_FAILED))
			{
				LeaveCS();
				FreeMutex();
				nPool=-1;
				return(false);
			}
			if (EnableCount) CptRetry++;
		}
		if (ThreadPoolRequested[nPool])
		{
			LeaveCS();
			FreeMutex();
			nPool=-1;
			return(false);
		}
	}
	else
	{
		bool PoolFree=true;
		
		for(uint8_t i=0; i<NbrePool; i++)
			PoolFree=PoolFree && (!ThreadPoolRequested[i]);
		
		// 一時的にHANDLE配列を作成してWaitForMultipleObjectsを呼び出す
		HANDLE tempHandles[MAX_THREAD_POOL];
		for(uint8_t i=0; i<NbrePool; i++) {
			tempHandles[i] = ThreadPoolFree[i].get();
		}

		while ((!PoolFree) && (CptRetry<NbreMaxRetry))
		{
			LeaveCS();
			resultWait=WaitForMultipleObjects(NbrePool, tempHandles, TRUE, TimeOutWait);
			EnterCS();
			userindex=GetUserIdIndex(UserId);
			if ((!Status_Ok) || Error_Occured || (userindex==-1)
				|| (resultWait==WAIT_TIMEOUT) || (resultWait==WAIT_FAILED))
			{
				LeaveCS();
				FreeMutex();
				nPool=-1;
				return(false);
			}
			if (EnableCount) CptRetry++;
			PoolFree=true;
			for(uint8_t i=0; i<NbrePool; i++)
				PoolFree=PoolFree && (!ThreadPoolRequested[i]);
		}
		if (!PoolFree)
		{
			LeaveCS();
			FreeMutex();
			nPool=-1;
			return(false);
		}
	}
	
	bool out=ptrPool[nPool]->RequestThreadPool(thread_number,Data,priority);
	
	if (out)
	{
		ThreadPoolRequested[nPool]=true;
		ResetEvent(ThreadPoolFree[nPool].get());
		if (Exclusive)
		{
			ExclusiveMode=true;
			ResetEvent(EndExclusive.get());
		}
		ThreadPoolUserId[nPool]=UserId;

		int8_t i=0;

		while ((i<MAX_THREAD_POOL) && (TabId[userindex].UsedPool[i]!=-1)) i++;
		if (i<MAX_THREAD_POOL)
		{
			idxPool=i;
			TabId[userindex].UsedPool[i]=nPool;
			TabId[userindex].NbrePool++;
		}
	}
	else nPool=-1;
	
	LeaveCS();
	FreeMutex();

	return(out);	
}


bool ThreadPoolInterface::ReleaseThreadPool(uint32_t UserId,bool sleep)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;

	for (int8_t i=0; i<MAX_THREAD_POOL; i++)
	{
		int8_t nPool=TabId[index].UsedPool[i];

		if ((nPool>=0) && (nPool<(int8_t)NbrePool) && (ThreadPoolUserId[nPool]==UserId))
			ok=ok && ReleaseThreadPoolCore(UserId,index,sleep,nPool,i);

		index=GetUserIdIndex(UserId);
		if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0))
		{
			ok=false;
			break;
		}
	}

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::ReleaseThreadPool(uint32_t UserId,bool sleep,int8_t idxPool)
{
	if ((!Status_Ok) || (UserId==0) || (idxPool<0) || (idxPool>=MAX_THREAD_POOL)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0))
	{
		LeaveCS();
		return(false);
	}

	int8_t nPool=TabId[index].UsedPool[idxPool];

	if ((nPool<-1) || (nPool>=(int8_t)NbrePool) || (ThreadPoolUserId[nPool]!=UserId))
	{
		LeaveCS();
		return(false);
	}

	if ((nPool==-1) || (ThreadPoolUserId[nPool]==0))
	{
		LeaveCS();
		return(true);
	}

	bool ok=ReleaseThreadPoolCore(UserId,index,sleep,nPool,idxPool);

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::ReleaseThreadPoolCore(uint32_t UserId,int32_t index,bool sleep,int8_t nPool,int8_t idxPool)
{
	bool out=true;

	ThreadPoolReleased[nPool]=true;

	if (ThreadPoolRequested[nPool])
	{
		while (JobsRunning[nPool])
		{
			HANDLE h=JobsEnded[nPool].get();
			
			LeaveCS();
			WaitForSingleObject(h,INFINITE);
			EnterCS();
			index=GetUserIdIndex(UserId);
			if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0))
			{
				ThreadPoolReleased[nPool]=false;				
				return(false);
			}
		}
		ThreadPoolRequested[nPool]=false;
		out=ptrPool[nPool]->ReleaseThreadPool(sleep);
		SetEvent(ThreadPoolFree[nPool].get());
	}

	if (ExclusiveMode)
	{
		ExclusiveMode=false;
		SetEvent(EndExclusive.get());
	}

	ThreadPoolReleased[nPool]=false;
	ThreadPoolUserId[nPool]=0;
	TabId[index].NbrePool--;
	TabId[index].UsedPool[idxPool]=-1;

	return(out);
}


bool ThreadPoolInterface::StartThreads(uint32_t UserId)
{
	if ((!Status_Ok) || Error_Occured || (UserId==0)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (index==-1) || (TabId[index].NbrePool==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;

	for (int8_t i=0; i<MAX_THREAD_POOL; i++)
	{
		int8_t nPool=TabId[index].UsedPool[i];

		if ((nPool>=0) && (nPool<(int8_t)NbrePool) && (ThreadPoolUserId[nPool]==UserId))
			ok=ok && StartThreadsCore(nPool);
	}

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::StartThreads(uint32_t UserId,int8_t idxPool)
{
	if ((!Status_Ok) || Error_Occured || (UserId==0) || (idxPool<0) || (idxPool>=MAX_THREAD_POOL)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (index==-1) || (TabId[index].NbrePool==0))
	{
		LeaveCS();
		return(false);
	}

	int8_t nPool=TabId[index].UsedPool[idxPool];

	if ((nPool<0) || (nPool>=(int8_t)NbrePool) || (ThreadPoolUserId[nPool]!=UserId))
	{
		LeaveCS();
		return(false);
	}

	bool ok=StartThreadsCore(nPool);

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::StartThreadsCore(int8_t nPool)
{
	if ((!ThreadPoolRequested[nPool]) || ThreadPoolReleased[nPool] || ThreadWaitEnd[nPool] || ThreadPoolWaitFree[nPool]) return(false);

	if (JobsRunning[nPool]) return(true);

	bool out=ptrPool[nPool]->StartThreads();

	if (out)
	{
		JobsRunning[nPool]=true;
		ResetEvent(JobsEnded[nPool].get());
	}

	return(out);	
}


bool ThreadPoolInterface::WaitThreadsEnd(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;

	for (int8_t i=0; i<MAX_THREAD_POOL; i++)
	{
		int8_t nPool=TabId[index].UsedPool[i];

		if ((nPool>=0) && (nPool<(int8_t)NbrePool) && (ThreadPoolUserId[nPool]==UserId))
			ok=ok && WaitThreadsEndCore(UserId,nPool,i);

		index=GetUserIdIndex(UserId);
		if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0))
		{
			ok=false;
			break;
		}
	}

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::WaitThreadsEnd(uint32_t UserId,int8_t idxPool)
{
	if ((!Status_Ok) || (UserId==0) || (idxPool<0) || (idxPool>=MAX_THREAD_POOL)) return(false);

	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || Error_Occured || (index==-1))
	{
		LeaveCS();
		return(false);
	}

	if (TabId[index].NbrePool==0)
	{
		LeaveCS();
		return(false);
	}

	int8_t nPool=TabId[index].UsedPool[idxPool];

	if ((nPool<0) || (nPool>=(int8_t)NbrePool) || (ThreadPoolUserId[nPool]!=UserId))
	{
		LeaveCS();
		return(false);
	}

	bool ok=WaitThreadsEndCore(UserId,nPool,idxPool);

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::WaitThreadsEndCore(uint32_t UserId,int8_t nPool,int8_t idxPool)
{
	if ((!ThreadPoolRequested[nPool]) || ThreadWaitEnd[nPool]) return(false);

	if (!JobsRunning[nPool]) return(true);
	
	ThreadWaitEnd[nPool]=true;

	ThreadPool *ptr=ptrPool[nPool];
	
	LeaveCS();
	bool out=ptr->WaitThreadsEnd();
	EnterCS();

	int32_t index=GetUserIdIndex(UserId);

	if ((!Status_Ok) || (index==-1) || (TabId[index].NbrePool==0)) return(false);

	nPool=TabId[index].UsedPool[idxPool];

	if (((nPool<0) || (nPool>=(int8_t)NbrePool)) || (ThreadPoolUserId[nPool]!=UserId)) return(false);

	ThreadWaitEnd[nPool]=false;

	if (out)
	{
		JobsRunning[nPool]=false;
		SetEvent(JobsEnded[nPool].get());
	}

	return(out);
}


bool ThreadPoolInterface::GetThreadPoolStatus(uint32_t UserId,int8_t idxPool,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (TabId.size()==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetThreadPoolStatus());
			else return(false);
		}
		else
		{
			int32_t i=GetUserIdIndex(UserId);

			if ((i==-1) || (TabId[i].NbrePool==0) || ((idxPool<0) || (idxPool>=MAX_THREAD_POOL)))
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetThreadPoolStatus());
				else return(false);
			}
			else
			{
				if ((TabId[i].UsedPool[idxPool]>=0) && (TabId[i].UsedPool[idxPool]<(int8_t)NbrePool)
					&& (ThreadPoolUserId[TabId[i].UsedPool[idxPool]]==UserId))
					return(ptrPool[TabId[i].UsedPool[idxPool]]->GetThreadPoolStatus());
				else return(false);
			}
		}
	}
	else return(false);
}


uint8_t ThreadPoolInterface::GetCurrentThreadAllocated(uint32_t UserId,int8_t idxPool,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (TabId.size()==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadAllocated());
			else return(0);
		}
		else
		{
			int32_t i=GetUserIdIndex(UserId);

			if ((i==-1) || (TabId[i].NbrePool==0) || ((idxPool<0) || (idxPool>=MAX_THREAD_POOL)))
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadAllocated());
				else return(0);
			}
			else
			{
				if ((TabId[i].UsedPool[idxPool]>=0) && (TabId[i].UsedPool[idxPool]<(int8_t)NbrePool))
					return(ptrPool[TabId[i].UsedPool[idxPool]]->GetCurrentThreadAllocated());
				else return(0);
			}
		}
	}
	else return(0);
}


uint8_t ThreadPoolInterface::GetCurrentThreadUsed(uint32_t UserId,int8_t idxPool,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (TabId.size()==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadUsed());
			else return(0);
		}
		else
		{
			int32_t i=GetUserIdIndex(UserId);

			if ((i==-1) || (TabId[i].NbrePool==0) || ((idxPool<0) || (idxPool>=MAX_THREAD_POOL)))
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadUsed());
				else return(0);
			}
			else
			{
				if ((TabId[i].UsedPool[idxPool]>=0) && (TabId[i].UsedPool[idxPool]<(int8_t)NbrePool))
					return(ptrPool[TabId[i].UsedPool[idxPool]]->GetCurrentThreadUsed());
				else return(0);
			}
		}
	}
	else return(0);
}


bool ThreadPoolInterface::EnableAllowSeveral(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowSeveral=true;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::DisableAllowSeveral(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowSeveral=false;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::IsAllowedSeveral(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0) || (NbrePool==0) || (TabId.size()==0)) return(false);

	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) return(TabId[i].AllowSeveral);
	else return(false);
}


bool ThreadPoolInterface::EnableWaitonRequest(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowWaiting=true;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::DisableWaitonRequest(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowWaiting=false;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::EnableTimeOutonRequest(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowTimeOut=true;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::DisableTimeOutonRequest(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowTimeOut=false;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::EnableRetryMaxonRequest(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowRetryMax=true;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::DisableRetryMaxonRequest(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) TabId[i].AllowRetryMax=false;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::ConfigureTimeOutValue(uint32_t UserId, DWORD dwMilliseconds)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if ((i!=-1) && (dwMilliseconds!=INFINITE)) TabId[i].TimeOut=dwMilliseconds;
	else ok=false;

	LeaveCS();

	return(ok);
}


bool ThreadPoolInterface::ConfigureRetryMaxValue(uint32_t UserId, uint8_t NbreMax)
{
	if ((!Status_Ok) || (UserId==0)) return(false);

	EnterCS();

	if ((!Status_Ok) || (NbrePool==0) || (TabId.size()==0))
	{
		LeaveCS();
		return(false);
	}

	bool ok=true;
	int32_t i=GetUserIdIndex(UserId);

	if ((i!=-1) && (NbreMax!=0)) TabId[i].RetryMax=NbreMax;
	else ok=false;

	LeaveCS();

	return(ok);
}


int8_t ThreadPoolInterface::GetPoolAllocated(uint32_t UserId)
{
	if ((!Status_Ok) || (UserId==0) || (NbrePool==0) || (TabId.size()==0)) return(-1);

	int32_t i=GetUserIdIndex(UserId);

	if (i!=-1) return(TabId[i].NbrePool);
	else return(-1);
}


int8_t ThreadPoolInterface::GetPoolNumber(uint32_t UserId,int8_t idxPool)
{
	if ((!Status_Ok) || (UserId==0) || (NbrePool==0) || (TabId.size()==0) || (idxPool<0) || (idxPool>=MAX_THREAD_POOL)) return(-1);

	int32_t i=GetUserIdIndex(UserId);

	int8_t valret=-1;

	if (i!=-1)
	{
		int8_t nPool=TabId[i].UsedPool[idxPool];

		if ((nPool>=0) && (nPool<(int8_t)NbrePool)) valret=nPool;
	}

	return(valret);
}


int8_t ThreadPoolInterface::GetPoolIndex(uint32_t UserId,int8_t nPool)
{
	if ((!Status_Ok) || (UserId==0) || (NbrePool==0) || (TabId.size()==0) || (nPool<0) || (nPool>=(int8_t)NbrePool)) return(-1);

	int32_t i=GetUserIdIndex(UserId);

	int8_t valret=-1;

	if (i!=-1)
	{
		if (TabId[i].NbrePool>0)
		{
			int8_t j=0;

			while ((j<MAX_THREAD_POOL) && (TabId[i].UsedPool[j]!=nPool)) j++;

			if (j<MAX_THREAD_POOL) valret=j;
		}
	}

	return(valret);
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

