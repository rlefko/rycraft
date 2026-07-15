#pragma once

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

enum class ThreadPriority {
    USER_INITIATED,
    UTILITY,
    BACKGROUND,
};

inline void setCurrentThreadPriority(ThreadPriority priority) {
#if defined(__APPLE__)
    qos_class_t qosClass = QOS_CLASS_UTILITY;
    switch (priority) {
        case ThreadPriority::USER_INITIATED:
            qosClass = QOS_CLASS_USER_INITIATED;
            break;
        case ThreadPriority::UTILITY:
            qosClass = QOS_CLASS_UTILITY;
            break;
        case ThreadPriority::BACKGROUND:
            qosClass = QOS_CLASS_BACKGROUND;
            break;
    }
    (void)pthread_set_qos_class_self_np(qosClass, 0);
#else
    (void)priority;
#endif
}
