/** @file
 * @brief Strict JSON parsing and canonical serialization.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_PROTOCOL_JSON_H
#define SS_PROTOCOL_JSON_H

#include <stddef.h>

/** Outcome of strict JSON canonicalization. */
typedef enum {
  SS_JSON_OK = 0,           /**< The input is valid strict JSON. */
  SS_JSON_INVALID_JSON,     /**< Syntax error. */
  SS_JSON_DUPLICATE_KEY,    /**< An object repeats a key. */
  SS_JSON_INVALID_NUMBER,   /**< Malformed, out-of-range or nonfinite number. */
  SS_JSON_AMBIGUOUS_NUMBER, /**< Negative zero, which has no canonical form. */
  SS_JSON_INVALID_UNICODE,  /**< Invalid UTF-8 or escape sequence. */
  SS_JSON_TRAILING_DATA,    /**< Data after the top-level value. */
  SS_JSON_OUT_OF_MEMORY     /**< Allocation failure. */
} SsJsonStatus;

/** Failure of a strict JSON operation. */
typedef struct {
  SsJsonStatus status; /**< Failure kind. */
  size_t offset;       /**< Byte offset of the failure in the input. */
} SsJsonError;

/**
 * Parse strict JSON and serialize it canonically.
 *
 * The canonical form sorts object keys and removes insignificant white
 * space; content identities are SHA-256 digests of it. Thread-safe.
 *
 * @param[in] input UTF-8 JSON text.
 * @param[in] input_size Byte count of @p input.
 * @param[out] output Canonical text, owned by the caller and released with
 *   ss_json_free(); NULL on failure.
 * @param[out] output_size Byte count of @p output.
 * @param[out] error Failure kind and offset; may be NULL.
 * @return ::SS_JSON_OK or the first violation found.
 */
SsJsonStatus ss_json_canonicalize(const unsigned char *input, size_t input_size,
                                  unsigned char **output, size_t *output_size,
                                  SsJsonError *error);

/** Stable name of a JSON status.
 * @param[in] status Status to name.
 * @return Static lowercase name such as `duplicate_key`. */
const char *ss_json_status_name(SsJsonStatus status);

/** Release a buffer returned by ss_json_canonicalize().
 * @param[in] pointer Buffer to release; NULL is accepted. */
void ss_json_free(void *pointer);

#endif
