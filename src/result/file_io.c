/* File staging, exclusive writes, synchronization, and publication.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_internal.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

SsStatus ss_result_payload_digest(const char *path, char digest[SS_SHA256_HEX_CAPACITY],
                                  uint64_t *bytes) {
  return ss_sha256_file(path, digest, bytes);
}

static SsStatus ss_result_fsync_close(int descriptor) {
  int failed = fsync(descriptor) != 0;
  if (close(descriptor) != 0) {
    failed = 1;
  }
  return failed != 0 ? SS_ERROR_IO : SS_OK;
}

static SsStatus ss_result_fsync_object(const char *path) {
  struct stat metadata;
  int descriptor;
  if (path == NULL || lstat(path, &metadata) != 0 ||
      (!S_ISREG(metadata.st_mode) && !S_ISDIR(metadata.st_mode))) {
    return SS_ERROR_IO;
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC);
  return descriptor < 0 ? SS_ERROR_IO : ss_result_fsync_close(descriptor);
}

SsStatus ss_result_fsync_directory(const char *path) {
  int descriptor = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  return descriptor < 0 ? SS_ERROR_IO : ss_result_fsync_close(descriptor);
}

static SsStatus ss_result_fsync_parent(const char *path) {
  char parent[SS_RESULT_PATH_CAPACITY];
  char name[SS_RESULT_TEXT_CAPACITY];
  SsStatus status =
      ss_result_path_split(path, parent, sizeof(parent), name, sizeof(name));
  if (status != SS_OK) {
    return status;
  }
  return ss_result_fsync_object(parent);
}

static SsStatus ss_result_write_exclusive(const char *path, const void *data,
                                          size_t size, int remove_partial) {
  const unsigned char *bytes = (const unsigned char *)data;
  int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  size_t written = 0U;
  SsStatus status = SS_OK;
  if (descriptor < 0) {
    return errno == EEXIST ? SS_ERROR_DUPLICATE : SS_ERROR_IO;
  }
  while (written < size) {
    const ssize_t count = write(descriptor, bytes + written, size - written);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      status = SS_ERROR_IO;
      break;
    }
    written += (size_t)count;
  }
  if (status == SS_OK) {
    status = ss_result_fsync_close(descriptor);
  } else {
    (void)close(descriptor);
  }
  if (status != SS_OK && remove_partial != 0) {
    (void)unlink(path);
  }
  return status;
}

SsStatus ss_result_write_staged_file(const char *path, const void *data, size_t size) {
  return ss_result_write_exclusive(path, data, size, 0);
}

SsStatus ss_result_write_text_atomic(const char *path, const char *text, size_t size) {
  char temporary[SS_RESULT_PATH_CAPACITY];
  SsStatus status;
  if (path == NULL || text == NULL || size == 0U ||
      snprintf(temporary, sizeof(temporary), "%s.tmp", path) < 0 ||
      strlen(temporary) >= sizeof(temporary)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_result_write_exclusive(temporary, text, size, 1);
  if (status != SS_OK) {
    return status;
  }
  if (renameat2(AT_FDCWD, temporary, AT_FDCWD, path, RENAME_NOREPLACE) != 0) {
    const int saved_errno = errno;
    (void)unlink(temporary);
    return saved_errno == EEXIST || saved_errno == ENOTEMPTY ? SS_ERROR_DUPLICATE
                                                             : SS_ERROR_IO;
  }
  return ss_result_fsync_parent(path);
}

SsStatus ss_result_remove_staging(const char *path) {
  DIR *directory;
  struct dirent *entry;
  char child[SS_RESULT_PATH_CAPACITY];
  int failed = 0;
  if (path == NULL || strstr(path, "/.slope-") == NULL) {
    return SS_ERROR_INVALID_PATH;
  }
  directory = opendir(path);
  if (directory == NULL) {
    return errno == ENOENT ? SS_OK : SS_ERROR_IO;
  }
  while ((entry = readdir(directory)) != NULL) {
    struct stat metadata;
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) < 0 ||
        strlen(child) >= sizeof(child) || lstat(child, &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || unlink(child) != 0) {
      failed = 1;
    }
  }
  if (closedir(directory) != 0 || rmdir(path) != 0) {
    failed = 1;
  }
  return failed != 0 ? SS_ERROR_IO : SS_OK;
}

SsStatus ss_result_path_split(const char *final_path, char *parent,
                              size_t parent_capacity, char *name,
                              size_t name_capacity) {
  const char *separator;
  size_t parent_size;
  size_t name_size;
  if (final_path == NULL || parent == NULL || name == NULL || parent_capacity == 0U ||
      name_capacity == 0U || final_path[0] != '/') {
    return SS_ERROR_INVALID_PATH;
  }
  separator = strrchr(final_path, '/');
  if (separator == NULL || separator == final_path || separator[1] == '\0') {
    return SS_ERROR_INVALID_PATH;
  }
  parent_size = (size_t)(separator - final_path);
  name_size = strlen(separator + 1);
  if (parent_size >= parent_capacity || name_size >= name_capacity ||
      strcmp(separator + 1, ".") == 0 || strcmp(separator + 1, "..") == 0 ||
      strchr(separator + 1, '/') != NULL) {
    return SS_ERROR_INVALID_PATH;
  }
  memcpy(parent, final_path, parent_size);
  parent[parent_size] = '\0';
  memcpy(name, separator + 1, name_size + 1U);
  return SS_OK;
}

SsStatus ss_result_make_staging(MPI_Comm communicator, const char *final_path,
                                const char *kind, char *staging,
                                size_t staging_capacity) {
  char parent[SS_RESULT_PATH_CAPACITY];
  char name[SS_RESULT_TEXT_CAPACITY];
  struct stat metadata;
  int rank = -1;
  int status_value = (int)SS_OK;
  unsigned int attempt;
  if (staging == NULL || staging_capacity < SS_RESULT_PATH_CAPACITY ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(staging, 0, staging_capacity);
  if (rank == 0) {
    SsStatus status =
        ss_result_path_split(final_path, parent, sizeof(parent), name, sizeof(name));
    if (status == SS_OK &&
        (stat(parent, &metadata) != 0 || !S_ISDIR(metadata.st_mode))) {
      status = SS_ERROR_INVALID_PATH;
    }
    if (status == SS_OK && lstat(final_path, &metadata) == 0) {
      status = SS_ERROR_DUPLICATE;
    } else if (status == SS_OK && errno != ENOENT) {
      status = SS_ERROR_IO;
    }
    for (attempt = 0U; status == SS_OK && attempt < 64U; ++attempt) {
      struct timespec now;
      int count;
      if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
        status = SS_ERROR_IO;
        break;
      }
      count = snprintf(staging, staging_capacity, "%s/.slope-%s-%ld-%lld-%u", parent,
                       kind, (long)getpid(), (long long)now.tv_nsec, attempt);
      if (count < 0 || (size_t)count >= staging_capacity) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      if (mkdir(staging, 0700) == 0) {
        break;
      }
      if (errno != EEXIST) {
        status = SS_ERROR_IO;
        break;
      }
      staging[0] = '\0';
    }
    if (status == SS_OK && staging[0] == '\0') {
      status = SS_ERROR_RESOURCE;
    }
    status_value = (int)status;
  }
  if (MPI_Bcast(&status_value, 1, MPI_INT, 0, communicator) != MPI_SUCCESS ||
      MPI_Bcast(staging, (int)staging_capacity, MPI_CHAR, 0, communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)status_value;
}

SsStatus ss_result_rank_zero_rename(MPI_Comm communicator, const char *source,
                                    const char *destination) {
  int rank = -1;
  int status_value = (int)SS_OK;
  if (source == NULL || destination == NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (rank == 0) {
    SsStatus status = ss_result_fsync_object(source);
    if (status == SS_OK) {
      status = ss_result_fsync_parent(source);
    }
    if (status == SS_OK &&
        renameat2(AT_FDCWD, source, AT_FDCWD, destination, RENAME_NOREPLACE) != 0) {
      status = errno == EEXIST || errno == ENOTEMPTY ? SS_ERROR_DUPLICATE : SS_ERROR_IO;
    }
    if (status == SS_OK) {
      status = ss_result_fsync_parent(destination);
    }
    status_value = (int)status;
  }
  if (MPI_Bcast(&status_value, 1, MPI_INT, 0, communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)status_value;
}
