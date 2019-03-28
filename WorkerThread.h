/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <condition_variable>
#include <future>
#include <deque>
#include <thread>
#include <pthread.h>

namespace faiss {

template<class Job>
struct Thread {
  Job job;
  pthread_t thread;

  Thread () {}

  explicit Thread (const Job & job): job(job) {}

  void start () {
    pthread_create (&thread, nullptr, run, this);
  }

  void wait () {
    pthread_join (thread, nullptr);
  }

  static void * run (void *arg) {
    static_cast<Thread*> (arg)->job.run();
    return nullptr;
  }
};


class WorkerThread {
 public:
  WorkerThread();

  /// Stops and waits for the worker thread to exit, flushing all
  /// pending lambdas
  ~WorkerThread();

  /// Request that the worker thread stop itself
  void stop();

  /// Blocking waits in the current thread for the worker thread to
  /// stop
  void waitForThreadExit();

  /// Adds a lambda to run on the worker thread; returns a future that
  /// can be used to block on its completion.
  /// Future status is `true` if the lambda was run in the worker
  /// thread; `false` if it was not run, because the worker thread is
  /// exiting or has exited.
  std::future<bool> add(std::function<void()> f);

 private:
  void startThread();
  void threadMain();
  void threadLoop();

  /// Thread that all queued lambdas are run on
  std::thread thread_;

  /// Mutex for the queue and exit status
  std::mutex mutex_;

  /// Monitor for the exit status and the queue
  std::condition_variable monitor_;

  /// Whether or not we want the thread to exit
  bool wantStop_;

  /// Queue of pending lambdas to call
  std::deque<std::pair<std::function<void()>, std::promise<bool>>> queue_;
};

} // namespace
