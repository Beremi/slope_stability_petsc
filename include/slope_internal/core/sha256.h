/** @file
 * @brief SHA-256 over bytes, files and incrementally supplied data.
 *
 * A context is owned by one thread; distinct contexts may be used
 * concurrently.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_SHA256_H
#define SS_CORE_SHA256_H

#include "slope_internal/core/status.h"

#include <stddef.h>
#include <stdint.h>

/** Size of a raw SHA-256 digest in bytes. */
#define SS_SHA256_SIZE 32U
/** Capacity of a hexadecimal digest including the terminator. */
#define SS_SHA256_HEX_CAPACITY (2U * SS_SHA256_SIZE + 1U)

/** Incremental SHA-256 context. */
typedef struct {
  uint32_t state[8];       /**< Chaining state. */
  uint64_t bit_count;      /**< Number of bits hashed so far. */
  unsigned char block[64]; /**< Pending partial block. */
  size_t block_size;       /**< Bytes in @ref block. */
} SsSha256;

/** Initialize or reset a caller-owned context.
 * @param[out] context Context to initialize. */
void ss_sha256_init(SsSha256 *context);

/** Append bytes to the hashed message.
 * @param[in,out] context Initialized context.
 * @param[in] data @p size bytes; NULL is allowed when @p size is zero.
 * @param[in] size Number of bytes.
 * @return ::SS_OK, or ::SS_ERROR_OVERFLOW for invalid arguments or a
 *   message longer than the SHA-256 limit. */
SsStatus ss_sha256_update(SsSha256 *context, const void *data, size_t size);

/** Finish the message and write its lowercase hexadecimal digest.
 * @param[in,out] context Context; it must be reinitialized before reuse.
 * @param[out] output NUL-terminated digest of 64 characters. */
void ss_sha256_final(SsSha256 *context, char output[SS_SHA256_HEX_CAPACITY]);

/** Finish the message and write its raw digest.
 * @param[in,out] context Context; it must be reinitialized before reuse.
 * @param[out] output 32-byte digest. */
void ss_sha256_final_bytes(SsSha256 *context, unsigned char output[SS_SHA256_SIZE]);

/** Hash a contiguous byte range.
 * @param[in] data @p size bytes; NULL is allowed when @p size is zero.
 * @param[in] size Number of bytes.
 * @param[out] output NUL-terminated hexadecimal digest.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_OVERFLOW. */
SsStatus ss_sha256_bytes(const unsigned char *data, size_t size,
                         char output[SS_SHA256_HEX_CAPACITY]);
/** Hash an open file from its current position to its end, without closing it.
 * @param[in] descriptor Readable file descriptor.
 * @param[out] output NUL-terminated hexadecimal digest.
 * @param[out] byte_count Number of bytes hashed.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_IO or
 *   ::SS_ERROR_OVERFLOW. */
SsStatus ss_sha256_fd(int descriptor, char output[SS_SHA256_HEX_CAPACITY],
                      uint64_t *byte_count);
/** Open, hash and close a file.
 * @param[in] path File to hash.
 * @param[out] output NUL-terminated hexadecimal digest.
 * @param[out] byte_count File size in bytes.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_IO or
 *   ::SS_ERROR_OVERFLOW. */
SsStatus ss_sha256_file(const char *path, char output[SS_SHA256_HEX_CAPACITY],
                        uint64_t *byte_count);

#endif
