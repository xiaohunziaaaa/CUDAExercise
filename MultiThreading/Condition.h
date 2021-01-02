#ifndef __CONDITION_H__
#define __CONDITION_H__                           
#include "MutexLockGuard.h"
#include <pthread.h>
class Condition{
    private:
    pthread_cond_t cond;
    MutexLock& mutex;

    public:
    Condition(MutexLock& _mutex):mutex(_mutex){     
        pthread_cond_init(&cond, NULL);
    };
    ~Condition(){
        pthread_cond_destroy(&cond);
    };

    void wait(){
        pthread_cond_wait(&cond, mutex.getmutex());
    };
    void notify(){
        pthread_cond_signal(&cond);
    };
    void notifyAll(){
        pthread_cond_broadcast(&cond);
    };

};


#endif