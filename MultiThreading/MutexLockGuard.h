#ifndef __MUTEXLOCKGUARD_H__
#define __MUTEXLOCKGUARD_H__
#include "MutexLock.h"

class MutexLockGuard{
        private:
        MutexLock& mutex;

        public:
        explicit MutexLockGuard(MutexLock& _mutex):mutex(_mutex){
            mutex.lock();
        }

        ~MutexLockGuard(){
            mutex.unlock();
        }
};
#endif