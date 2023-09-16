#ifndef LOGGING_LOG_H
#define LOGGING_LOG_H

#include <glog/logging.h>

#define MLOG_ERROR_FILE(file, message)                                         \
  google::InitGoogleLogging(file),                                             \
      LOG(ERROR) << "\033[31m"                                                 \
                 << "[" << __FILE__ << __LINE__ << "]: " << message            \
                 << "\033[0m",                                                 \
      google::ShutdownGoogleLogging();

#define MLOG_WARNING_FILE(file, message)                                       \
  google::InitGoogleLogging(file),                                             \
      LOG(WARNING) << "\033[32m"                                               \
                   << "[" << __FILE__ << __LINE__ << "]: " << message          \
                   << "\033[0m",                                               \
      google::ShutdownGoogleLogging();

#define MLOG_FATAL_FILE(file, message)                                         \
  google::InitGoogleLogging(file),                                             \
      LOG(FATAL) << "\033[33m"                                                 \
                 << "[" << __FILE__ << __LINE__ << "]: " << message            \
                 << "\033[0m",                                                 \
      google::ShutdownGoogleLogging();

#define MLOG_INFO_FILE(file, message)                                          \
  google::InitGoogleLogging(file),                                             \
      LOG(INFO) << "\033[95m"                                                  \
                << "[" << __FILE__ << __LINE__ << "]: " << message             \
                << "\033[0m",                                                  \
      google::ShutdownGoogleLogging();

#define MLOG_ERROR(message)                                                    \
  google::InitGoogleLogging("log"),                                            \
      LOG(ERROR) << "\033[31m"                                                 \
                 << "[" << __FILE__ << __LINE__ << "]: " << message            \
                 << "\033[0m",                                                 \
      google::ShutdownGoogleLogging();

#define MLOG_WARNING(message)                                                  \
  google::InitGoogleLogging("log"),                                            \
      LOG(WARNING) << "\033[32m"                                               \
                   << "[" << __FILE__ << __LINE__ << "]: " << message          \
                   << "\033[0m",                                               \
      google::ShutdownGoogleLogging();

#define MLOG_FATAL(message)                                                    \
  google::InitGoogleLogging("log"),                                            \
      LOG(FATAL) << "\033[33m"                                                 \
                 << "[" << __FILE__ << __LINE__ << "]: " << message            \
                 << "\033[0m",                                                 \
      google::ShutdownGoogleLogging();

#define MLOG_INFO(message)                                                     \
  google::InitGoogleLogging("log"),                                            \
      LOG(INFO) << "\033[95m"                                                  \
                << "[" << __FILE__ << __LINE__ << "]: " << message             \
                << "\033[0m",                                                  \
      google::ShutdownGoogleLogging();

#endif // LOGGING_LOG_H
