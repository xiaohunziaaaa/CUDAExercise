#ifndef __MUTEXLOCK_H__
#define __MUTEXLOCK_H__
#include <pthread.h>

class MutexLock{
    private:
    pthread_mutex_t mutex;

    public:
    MutexLock(){
        pthread_mutex_init(&mutex, NULL);
    }

    ~MutexLock(){
        pthread_mutex_destroy(&mutex);
    }

    void lock(){
        pthread_mutex_lock(&mutex);
    }

    void unlock(){
        pthread_mutex_unlock(&mutex);
    }

    pthread_mutex_t* getmutex(){
        return &mutex;
    }
};
#endif