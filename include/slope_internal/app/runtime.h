/** @file
 * @brief Install identity, result directory preparation and atomic JSON
 * publication.
 *
 * These functions are local to the calling rank; the engine calls them on
 * rank zero and agrees on the status.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_APP_RUNTIME_H
#define SS_APP_RUNTIME_H

#include "slope_internal/core/core.h"

#include <stddef.h>

/** Identity of the installed executable, verified against its manifest. */
typedef struct {
  /** `sha256:` identity of the executable projection of the install manifest. */
  char executable_id[72];
  /** Lowercase hexadecimal SHA-256 of the running engine executable. */
  char engine_sha256[SS_SHA256_HEX_CAPACITY];
  /** `sha256:` digest of the installed schema set. */
  char schema_id[72];
  /** Path of `share/slope-stability/install-manifest.json`. */
  char manifest_path[512];
  /** Resolved path of the running executable. */
  char executable_path[512];
} SsInstallIdentity;

/**
 * Locate the install manifest next to the executable and verify the install.
 *
 * The SHA-256 of the engine and of `slope-probe` must match the manifest, and
 * the executable identity is recomputed from the manifest projection.
 *
 * @param[in] argv0 `argv[0]` of the engine, resolved when `/proc/self/exe`
 *   cannot be read.
 * @param[out] identity Verified identity; zeroed first.
 * @param[out] error Optional error record describing a failure.
 * @return ::SS_OK; ::SS_ERROR_NOT_FOUND or ::SS_ERROR_IO for a missing file,
 *   ::SS_ERROR_SCHEMA for an invalid manifest, ::SS_ERROR_INTEGRITY for a
 *   digest or identity mismatch, ::SS_ERROR_LIMIT for an overlong path.
 */
SsStatus ss_install_identity_load(const char *argv0, SsInstallIdentity *identity,
                                  SsErrorRecord *error);

/**
 * Create the result directory, or accept an existing empty one.
 *
 * @param[in] path Directory path; it is created with mode 0700 when missing.
 * @param[out] error Optional error record describing a failure.
 * @return ::SS_OK; ::SS_ERROR_INVALID_PATH when the path is not a plain
 *   directory, ::SS_ERROR_INTEGRITY when it is not empty, ::SS_ERROR_IO when it
 *   cannot be created or read.
 */
SsStatus ss_prepare_result_directory(const char *path, SsErrorRecord *error);

/**
 * Publish a JSON document under its content identity.
 *
 * The body is canonicalized and hashed; the document is written with an added
 * `"id": "sha256:<hex>"` member to a temporary file, synchronized and renamed
 * into @p directory, so readers never observe a partial file.
 *
 * @param[in] directory Existing destination directory.
 * @param[in] filename File name inside @p directory.
 * @param[in] body_without_id JSON object text without its `id` member.
 * @param[out] output_id Content identity written as `sha256:<hex>`.
 * @param[out] error Optional error record describing a failure.
 * @return ::SS_OK; ::SS_ERROR_JSON for an invalid body, ::SS_ERROR_IO for a
 *   write, synchronization or rename failure.
 */
SsStatus ss_publish_content_json(const char *directory, const char *filename,
                                 const char *body_without_id, char output_id[72],
                                 SsErrorRecord *error);

#endif
