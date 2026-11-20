/* Entry point of slope-schema-check: strict parsing and canonical output
 * of a JSON document.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/protocol/json.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_file(const char *path, size_t *size) {
  FILE *stream;
  long end;
  unsigned char *data;
  size_t count;
  stream = fopen(path, "rb");
  if (stream == NULL) {
    return NULL;
  }
  if (fseek(stream, 0L, SEEK_END) != 0) {
    fclose(stream);
    return NULL;
  }
  end = ftell(stream);
  if (end < 0L || fseek(stream, 0L, SEEK_SET) != 0) {
    fclose(stream);
    return NULL;
  }
  *size = (size_t)end;
  data = (unsigned char *)malloc(*size == 0U ? 1U : *size);
  if (data == NULL) {
    fclose(stream);
    return NULL;
  }
  count = fread(data, 1U, *size, stream);
  if (count != *size || fclose(stream) != 0) {
    free(data);
    return NULL;
  }
  return data;
}

int main(int argc, char **argv) {
  unsigned char *input;
  unsigned char *output = NULL;
  size_t input_size = 0U;
  size_t output_size = 0U;
  SsJsonError error = {SS_JSON_OK, 0U};
  SsJsonStatus status;
  if (setlocale(LC_NUMERIC, "C") == NULL) {
    fputs("failed to select C numeric locale\n", stderr);
    return 70;
  }
  if (argc != 3 || strcmp(argv[1], "canonicalize") != 0) {
    fprintf(stderr, "usage: %s canonicalize FILE\n", argv[0]);
    return 64;
  }
  input = read_file(argv[2], &input_size);
  if (input == NULL) {
    fprintf(stderr, "cannot read %s\n", argv[2]);
    return 3;
  }
  status = ss_json_canonicalize(input, input_size, &output, &output_size, &error);
  free(input);
  if (status != SS_JSON_OK) {
    printf("{\"error\":\"%s\",\"offset\":%zu}\n", ss_json_status_name(status),
           error.offset);
    return 2;
  }
  if (fwrite(output, 1U, output_size, stdout) != output_size ||
      fputc('\n', stdout) == EOF) {
    ss_json_free(output);
    return 3;
  }
  ss_json_free(output);
  return 0;
}
