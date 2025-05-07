#ifndef __ThreadPoolInterface_H__
#define __ThreadPoolInterface_H__

#include "rgy_osdep.h"
#include "ThreadPoolDef.h"
#include "ThreadPool.h"

#define THREADPOOLINTERFACE_VERSION "ThreadPoolInterface 2.0.0"

typedef struct _UserData
{
	volatile uint16_t UserId;
	volatile int8_t nPool;
	volatile bool nPollTab[MAX_THREAD_POOL];
} UserData;

#if defined(_WIN32) || defined(_WIN64)
class ThreadPoolInterfaceWin;
typedef ThreadPoolInterfaceWin ThreadPoolInterface;
#else
class ThreadPoolInterfaceLinux;
typedef ThreadPoolInterfaceLinux ThreadPoolInterface;
#endif

// 基本インターフェースクラス
class ThreadPoolInterfaceBase
{
public:
	virtual ~ThreadPoolInterfaceBase(void);
	static ThreadPoolInterfaceBase* Init(uint8_t num);

	uint8_t GetThreadNumber(uint8_t thread_number, bool logical);
	int16_t AddPool(uint8_t num);
	bool CreatePool(uint8_t num);
	bool DeletePool(uint8_t num);
	bool RemovePool(uint8_t num);	
	bool AllocateThreads(uint8_t thread_number, uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep, int8_t nPool);
	bool GetUserId(uint16_t &UserId);
	bool RemoveUserId(uint16_t UserId);
	bool ChangeThreadsAffinity(uint8_t offset_core, uint8_t offset_ht, bool UseMaxPhysCore, bool SetAffinity, bool sleep, int8_t nPool);
	bool DeAllocateUserThreads(uint16_t UserId, bool check);
	bool DeAllocatePoolThreads(uint8_t nPool, bool check);
	bool DeAllocateAllThreads(bool check);
	bool RequestThreadPool(uint16_t UserId, uint8_t thread_number, Public_MT_Data_Thread *Data, int8_t nPool, bool Exclusive);
	bool RequestThreadPool(uint16_t UserId, uint8_t thread_number, Public_MT_Data_Thread *Data, int8_t &nPool, bool Exclusive, bool AllowSeveral);
	bool ReleaseThreadPool(uint16_t UserId, bool sleep);
	bool ReleaseThreadPool(uint16_t UserId, bool sleep, int8_t nPool);
	bool StartThreads(uint16_t UserId);
	bool StartThreads(uint16_t UserId, int8_t nPool);
	bool WaitThreadsEnd(uint16_t UserId);
	bool WaitThreadsEnd(uint16_t UserId, int8_t nPool);
	bool GetThreadPoolStatus(uint16_t UserId, int8_t nPool);
	uint8_t GetCurrentThreadAllocated(uint16_t UserId, int8_t nPool);
	uint8_t GetCurrentThreadUsed(uint16_t UserId, int8_t nPool);
	uint8_t GetLogicalCPUNumber(void);
	uint8_t GetPhysicalCoreNumber(void);
	
	bool GetThreadPoolInterfaceStatus(void) {return(Status_Ok);}
	int8_t GetCurrentPoolCreated(void) {return((Status_Ok) ? NbrePool:-1);}

#if defined(_WIN32) || defined(_WIN64)
	friend class ThreadPoolInterfaceWin;
#else
	friend class ThreadPoolInterfaceLinux;
#endif

protected:
	volatile bool Status_Ok;
	volatile uint8_t NbrePool;
	volatile uint16_t NbreUsers;
	volatile bool Error_Occured;
	UserData *TabId;
	
	ThreadPoolBase *ptrPool[MAX_THREAD_POOL];
	
	ThreadPoolInterfaceBase(void);
	
	virtual bool EnterCS(void) = 0;
	virtual void LeaveCS(void) = 0;
	virtual bool GetMutex(void) = 0;
	virtual void FreeMutex(void) = 0;
	virtual bool CreatePoolEvent(uint8_t num) = 0;
	virtual void FreeData(void) = 0;
	virtual void FreePool(void) = 0;
	virtual void FreePool(int8_t nPool) = 0;
	
	int16_t GetUserIdIndex(uint16_t UserId);
	bool ReleaseThreadPoolCore(uint16_t UserId, int16_t index, bool sleep, int8_t nPool);
	bool StartThreadsCore(int8_t nPool);
	bool WaitThreadsEndCore(uint16_t UserId, int8_t nPool);
	
	// OS依存の変数
	volatile bool ExclusiveMode;
	volatile bool *ThreadPoolRequested;
	volatile bool *JobsRunning;
	volatile bool *ThreadPoolReleased;
	volatile bool *ThreadWaitEnd;
	volatile bool *ThreadPoolWaitFree;
	volatile uint16_t *ThreadPoolUserId;
	
private:
	ThreadPoolInterfaceBase(const ThreadPoolInterfaceBase &other);
	ThreadPoolInterfaceBase& operator = (const ThreadPoolInterfaceBase &other);
	bool operator == (const ThreadPoolInterfaceBase &other) const;
	bool operator != (const ThreadPoolInterfaceBase &other) const;
};

#endif // __ThreadPoolInterface_H__

