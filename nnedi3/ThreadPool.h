#ifndef __ThreadPool_H__
#define __ThreadPool_H__

#include "rgy_osdep.h"
#include "ThreadPoolBase.h"
#include "ThreadPoolWin.h"
#include "ThreadPoolLinux.h"

#define THREADPOOL_VERSION "ThreadPool 2.0.0"

#if defined(_WIN32) || defined(_WIN64)
typedef ThreadPoolWin ThreadPool;
#else
typedef ThreadPoolLinux ThreadPool;
#endif

#endif // __ThreadPool_H__
