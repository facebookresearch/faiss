/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_TEST_UTIL_H
#define FAISS_TEST_UTIL_H

#include <faiss/IndexIVFPQ.h>
#include <unistd.h>

struct Tempfilename {
    pthread_mutex_t* mutex;
    std::string filename;

    Tempfilename(pthread_mutex_t* mutex, std::string filename_template) {
        this->mutex = mutex;
        this->filename = filename_template;
        pthread_mutex_lock(mutex);
        int fd = mkstemp(&this->filename[0]);
        close(fd);
        pthread_mutex_unlock(mutex);
    }

    ~Tempfilename() {
        if (access(filename.c_str(), F_OK)) {
            unlink(filename.c_str());
        }
    }

    const char* c_str() {
        return filename.c_str();
    }
};

#endif // FAISS_TEST_UTIL_H
