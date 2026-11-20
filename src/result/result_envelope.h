/* Shared JSON envelope fields and canonical serialization.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef SS_RESULT_ENVELOPE_H
#define SS_RESULT_ENVELOPE_H

#include "slope_internal/core/status.h"
#include "yyjson.h"

/** Add the kind, identity and schema version members to an envelope. */
int ss_result_envelope_header(yyjson_mut_doc *document, yyjson_mut_val *root,
                              yyjson_mut_val *version, const char *kind, const char *id,
                              uint32_t minor);
/** Add an identity and logical digest object to an envelope. */
int ss_result_envelope_identity(yyjson_mut_doc *document, yyjson_mut_val *identity,
                                const char *id_key, const char *id,
                                const char *logical_digest);
/** Add the run, problem and build identities to an envelope. */
int ss_result_envelope_run(yyjson_mut_doc *document, yyjson_mut_val *run,
                           const char *run_id, const char *problem_id,
                           const char *build_id);
/** Add the payload path, size and digests to an envelope. */
int ss_result_envelope_payload(yyjson_mut_doc *document, yyjson_mut_val *payload,
                               const char *path, uint64_t bytes, const char *sha256,
                               const char *logical_digest);
/** Serialize an envelope canonically; the caller frees the output. */
SsStatus ss_result_envelope_serialize(yyjson_mut_doc *document, unsigned char **output,
                                      size_t *output_size);
/** Copy a string member of a JSON object into a buffer. */
int ss_result_json_string(yyjson_val *object, const char *key, char *output,
                          size_t capacity);

#endif
