/* SHA-256 implementation and bounded file hashing.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sha256.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const uint32_t ss_sha256_constants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

static uint32_t ss_rotr(uint32_t value, unsigned int shift) {
  return (value >> shift) | (value << (32U - shift));
}

static void ss_sha256_transform(SsSha256 *context, const unsigned char block[64]) {
  uint32_t words[64];
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t e;
  uint32_t f;
  uint32_t g;
  uint32_t h;
  size_t index;
  for (index = 0U; index < 16U; ++index) {
    size_t offset = index * 4U;
    words[index] = ((uint32_t)block[offset] << 24U) |
                   ((uint32_t)block[offset + 1U] << 16U) |
                   ((uint32_t)block[offset + 2U] << 8U) | (uint32_t)block[offset + 3U];
  }
  for (index = 16U; index < 64U; ++index) {
    uint32_t s0 = ss_rotr(words[index - 15U], 7U) ^ ss_rotr(words[index - 15U], 18U) ^
                  (words[index - 15U] >> 3U);
    uint32_t s1 = ss_rotr(words[index - 2U], 17U) ^ ss_rotr(words[index - 2U], 19U) ^
                  (words[index - 2U] >> 10U);
    words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
  }
  a = context->state[0];
  b = context->state[1];
  c = context->state[2];
  d = context->state[3];
  e = context->state[4];
  f = context->state[5];
  g = context->state[6];
  h = context->state[7];
  for (index = 0U; index < 64U; ++index) {
    uint32_t sum1 = ss_rotr(e, 6U) ^ ss_rotr(e, 11U) ^ ss_rotr(e, 25U);
    uint32_t choice = (e & f) ^ ((~e) & g);
    uint32_t temporary1 = h + sum1 + choice + ss_sha256_constants[index] + words[index];
    uint32_t sum0 = ss_rotr(a, 2U) ^ ss_rotr(a, 13U) ^ ss_rotr(a, 22U);
    uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  context->state[0] += a;
  context->state[1] += b;
  context->state[2] += c;
  context->state[3] += d;
  context->state[4] += e;
  context->state[5] += f;
  context->state[6] += g;
  context->state[7] += h;
}

void ss_sha256_init(SsSha256 *context) {
  static const uint32_t initial[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  memset(context, 0, sizeof(*context));
  memcpy(context->state, initial, sizeof(initial));
}

SsStatus ss_sha256_update(SsSha256 *context, const void *data, size_t size) {
  const unsigned char *bytes = data;
  size_t offset = 0U;
  if (context == NULL || (data == NULL && size != 0U) ||
      size > (UINT64_MAX - context->bit_count) / 8U) {
    return SS_ERROR_OVERFLOW;
  }
  context->bit_count += (uint64_t)size * 8U;
  while (offset < size) {
    size_t available = sizeof(context->block) - context->block_size;
    size_t remaining = size - offset;
    size_t copied = remaining < available ? remaining : available;
    memcpy(context->block + context->block_size, bytes + offset, copied);
    context->block_size += copied;
    offset += copied;
    if (context->block_size == sizeof(context->block)) {
      ss_sha256_transform(context, context->block);
      context->block_size = 0U;
    }
  }
  return SS_OK;
}

void ss_sha256_final_bytes(SsSha256 *context, unsigned char digest[SS_SHA256_SIZE]) {
  size_t index;
  context->block[context->block_size++] = 0x80U;
  if (context->block_size > 56U) {
    memset(context->block + context->block_size, 0,
           sizeof(context->block) - context->block_size);
    ss_sha256_transform(context, context->block);
    context->block_size = 0U;
  }
  memset(context->block + context->block_size, 0, 56U - context->block_size);
  for (index = 0U; index < 8U; ++index) {
    context->block[63U - index] = (unsigned char)(context->bit_count >> (index * 8U));
  }
  ss_sha256_transform(context, context->block);
  for (index = 0U; index < 8U; ++index) {
    digest[index * 4U] = (unsigned char)(context->state[index] >> 24U);
    digest[index * 4U + 1U] = (unsigned char)(context->state[index] >> 16U);
    digest[index * 4U + 2U] = (unsigned char)(context->state[index] >> 8U);
    digest[index * 4U + 3U] = (unsigned char)context->state[index];
  }
}

static void ss_sha256_hex(const unsigned char digest[32],
                          char output[SS_SHA256_HEX_CAPACITY]) {
  static const char digits[] = "0123456789abcdef";
  size_t index;
  for (index = 0U; index < 32U; ++index) {
    output[index * 2U] = digits[digest[index] >> 4U];
    output[index * 2U + 1U] = digits[digest[index] & 0x0fU];
  }
  output[64] = '\0';
}

void ss_sha256_final(SsSha256 *context, char output[SS_SHA256_HEX_CAPACITY]) {
  unsigned char digest[32];
  ss_sha256_final_bytes(context, digest);
  ss_sha256_hex(digest, output);
}

SsStatus ss_sha256_bytes(const unsigned char *data, size_t size,
                         char output[SS_SHA256_HEX_CAPACITY]) {
  SsSha256 context;
  SsStatus status;
  if (output == NULL || (data == NULL && size != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_sha256_init(&context);
  status = ss_sha256_update(&context, data, size);
  if (status != SS_OK) {
    return status;
  }
  ss_sha256_final(&context, output);
  return SS_OK;
}

SsStatus ss_sha256_fd(int descriptor, char output[SS_SHA256_HEX_CAPACITY],
                      uint64_t *byte_count) {
  SsSha256 context;
  unsigned char buffer[65536];
  uint64_t total = 0U;
  if (descriptor < 0 || output == NULL || byte_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_sha256_init(&context);
  for (;;) {
    ssize_t count = read(descriptor, buffer, sizeof(buffer));
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count < 0) {
      return SS_ERROR_IO;
    }
    if (count == 0) {
      break;
    }
    if ((uint64_t)count > UINT64_MAX - total ||
        ss_sha256_update(&context, buffer, (size_t)count) != SS_OK) {
      return SS_ERROR_OVERFLOW;
    }
    total += (uint64_t)count;
  }
  ss_sha256_final(&context, output);
  *byte_count = total;
  return SS_OK;
}

SsStatus ss_sha256_file(const char *path, char output[SS_SHA256_HEX_CAPACITY],
                        uint64_t *byte_count) {
  FILE *stream;
  SsSha256 context;
  unsigned char buffer[65536];
  uint64_t total = 0U;
  if (path == NULL || output == NULL || byte_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  stream = fopen(path, "rb");
  if (stream == NULL) {
    return errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_IO;
  }
  ss_sha256_init(&context);
  for (;;) {
    size_t count = fread(buffer, 1U, sizeof(buffer), stream);
    if (count != 0U) {
      if ((uint64_t)count > UINT64_MAX - total ||
          ss_sha256_update(&context, buffer, count) != SS_OK) {
        (void)fclose(stream);
        return SS_ERROR_OVERFLOW;
      }
      total += (uint64_t)count;
    }
    if (count < sizeof(buffer)) {
      if (ferror(stream) != 0) {
        (void)fclose(stream);
        return SS_ERROR_IO;
      }
      break;
    }
  }
  if (fclose(stream) != 0) {
    return SS_ERROR_IO;
  }
  ss_sha256_final(&context, output);
  *byte_count = total;
  return SS_OK;
}
