/* Strict JSON parsing and canonical serialization over yyjson.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/protocol/json.h"
#include "slope_internal/core/sort.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  unsigned char *data;
  size_t size;
  size_t capacity;
} SsBuffer;

typedef struct {
  unsigned char *data;
  size_t size;
} SsString;

typedef enum {
  SS_VALUE_NULL,
  SS_VALUE_BOOL,
  SS_VALUE_NUMBER,
  SS_VALUE_STRING,
  SS_VALUE_ARRAY,
  SS_VALUE_OBJECT
} SsValueType;

typedef struct SsValue SsValue;

typedef struct {
  SsString key;
  SsValue *value;
} SsMember;

struct SsValue {
  SsValueType type;
  union {
    int boolean;
    SsString number;
    SsString string;
    struct {
      SsValue **items;
      size_t count;
      size_t capacity;
    } array;
    struct {
      SsMember *members;
      size_t count;
    } object;
  } as;
};

typedef struct {
  const unsigned char *input;
  size_t size;
  size_t position;
  SsJsonError error;
} SsParser;

static void ss_set_error(SsParser *parser, SsJsonStatus status) {
  if (parser->error.status == SS_JSON_OK) {
    parser->error.status = status;
    parser->error.offset = parser->position;
  }
}

static int ss_buffer_reserve(SsBuffer *buffer, size_t extra) {
  size_t required;
  size_t capacity;
  unsigned char *data;
  if (extra > SIZE_MAX - buffer->size) {
    return 0;
  }
  required = buffer->size + extra;
  if (required <= buffer->capacity) {
    return 1;
  }
  capacity = buffer->capacity == 0U ? 64U : buffer->capacity;
  while (capacity < required) {
    if (capacity > SIZE_MAX / 2U) {
      capacity = required;
      break;
    }
    capacity *= 2U;
  }
  data = (unsigned char *)realloc(buffer->data, capacity);
  if (data == NULL) {
    return 0;
  }
  buffer->data = data;
  buffer->capacity = capacity;
  return 1;
}

static int ss_buffer_append(SsBuffer *buffer, const void *data, size_t size) {
  if (!ss_buffer_reserve(buffer, size)) {
    return 0;
  }
  if (size != 0U) {
    memcpy(buffer->data + buffer->size, data, size);
  }
  buffer->size += size;
  return 1;
}

static int ss_buffer_byte(SsBuffer *buffer, unsigned char byte) {
  return ss_buffer_append(buffer, &byte, 1U);
}

static void ss_string_destroy(SsString *string) {
  free(string->data);
  string->data = NULL;
  string->size = 0U;
}

static void ss_value_destroy(SsValue *value) {
  size_t index;
  if (value == NULL) {
    return;
  }
  switch (value->type) {
  case SS_VALUE_NUMBER:
    ss_string_destroy(&value->as.number);
    break;
  case SS_VALUE_STRING:
    ss_string_destroy(&value->as.string);
    break;
  case SS_VALUE_ARRAY:
    for (index = 0U; index < value->as.array.count; ++index) {
      ss_value_destroy(value->as.array.items[index]);
    }
    free(value->as.array.items);
    break;
  case SS_VALUE_OBJECT:
    for (index = 0U; index < value->as.object.count; ++index) {
      ss_string_destroy(&value->as.object.members[index].key);
      ss_value_destroy(value->as.object.members[index].value);
    }
    free(value->as.object.members);
    break;
  case SS_VALUE_NULL:
  case SS_VALUE_BOOL:
    break;
  }
  free(value);
}

static SsValue *ss_value_create(SsParser *parser, SsValueType type) {
  SsValue *value = (SsValue *)calloc(1U, sizeof(*value));
  if (value == NULL) {
    ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
    return NULL;
  }
  value->type = type;
  return value;
}

static void ss_skip_space(SsParser *parser) {
  while (parser->position < parser->size) {
    unsigned char byte = parser->input[parser->position];
    if (byte != (unsigned char)' ' && byte != (unsigned char)'\t' &&
        byte != (unsigned char)'\n' && byte != (unsigned char)'\r') {
      break;
    }
    ++parser->position;
  }
}

static int ss_hex_value(unsigned char byte) {
  if (byte >= (unsigned char)'0' && byte <= (unsigned char)'9') {
    return (int)(byte - (unsigned char)'0');
  }
  if (byte >= (unsigned char)'a' && byte <= (unsigned char)'f') {
    return 10 + (int)(byte - (unsigned char)'a');
  }
  if (byte >= (unsigned char)'A' && byte <= (unsigned char)'F') {
    return 10 + (int)(byte - (unsigned char)'A');
  }
  return -1;
}

static int ss_parse_hex4(SsParser *parser, uint32_t *value) {
  size_t index;
  uint32_t result = 0U;
  if (parser->size - parser->position < 4U) {
    ss_set_error(parser, SS_JSON_INVALID_UNICODE);
    return 0;
  }
  for (index = 0U; index < 4U; ++index) {
    int digit = ss_hex_value(parser->input[parser->position + index]);
    if (digit < 0) {
      parser->position += index;
      ss_set_error(parser, SS_JSON_INVALID_UNICODE);
      return 0;
    }
    result = result * 16U + (uint32_t)digit;
  }
  parser->position += 4U;
  *value = result;
  return 1;
}

static int ss_append_codepoint(SsBuffer *buffer, uint32_t codepoint) {
  unsigned char encoded[4];
  size_t size;
  if (codepoint <= 0x7fU) {
    encoded[0] = (unsigned char)codepoint;
    size = 1U;
  } else if (codepoint <= 0x7ffU) {
    encoded[0] = (unsigned char)(0xc0U | (codepoint >> 6U));
    encoded[1] = (unsigned char)(0x80U | (codepoint & 0x3fU));
    size = 2U;
  } else if (codepoint <= 0xffffU) {
    encoded[0] = (unsigned char)(0xe0U | (codepoint >> 12U));
    encoded[1] = (unsigned char)(0x80U | ((codepoint >> 6U) & 0x3fU));
    encoded[2] = (unsigned char)(0x80U | (codepoint & 0x3fU));
    size = 3U;
  } else {
    encoded[0] = (unsigned char)(0xf0U | (codepoint >> 18U));
    encoded[1] = (unsigned char)(0x80U | ((codepoint >> 12U) & 0x3fU));
    encoded[2] = (unsigned char)(0x80U | ((codepoint >> 6U) & 0x3fU));
    encoded[3] = (unsigned char)(0x80U | (codepoint & 0x3fU));
    size = 4U;
  }
  return ss_buffer_append(buffer, encoded, size);
}

static int ss_copy_utf8(SsParser *parser, SsBuffer *buffer) {
  unsigned char first;
  uint32_t codepoint;
  size_t width;
  size_t index;
  if (parser->position >= parser->size) {
    ss_set_error(parser, SS_JSON_INVALID_UNICODE);
    return 0;
  }
  first = parser->input[parser->position];
  if (first < 0x80U) {
    if (!ss_buffer_byte(buffer, first)) {
      ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
      return 0;
    }
    ++parser->position;
    return 1;
  }
  if (first >= 0xc2U && first <= 0xdfU) {
    width = 2U;
    codepoint = (uint32_t)(first & 0x1fU);
  } else if (first >= 0xe0U && first <= 0xefU) {
    width = 3U;
    codepoint = (uint32_t)(first & 0x0fU);
  } else if (first >= 0xf0U && first <= 0xf4U) {
    width = 4U;
    codepoint = (uint32_t)(first & 0x07U);
  } else {
    ss_set_error(parser, SS_JSON_INVALID_UNICODE);
    return 0;
  }
  if (parser->size - parser->position < width) {
    ss_set_error(parser, SS_JSON_INVALID_UNICODE);
    return 0;
  }
  for (index = 1U; index < width; ++index) {
    unsigned char next = parser->input[parser->position + index];
    if ((next & 0xc0U) != 0x80U) {
      parser->position += index;
      ss_set_error(parser, SS_JSON_INVALID_UNICODE);
      return 0;
    }
    codepoint = (codepoint << 6U) | (uint32_t)(next & 0x3fU);
  }
  if ((width == 3U && codepoint < 0x800U) || (width == 4U && codepoint < 0x10000U) ||
      (codepoint >= 0xd800U && codepoint <= 0xdfffU) || codepoint > 0x10ffffU) {
    ss_set_error(parser, SS_JSON_INVALID_UNICODE);
    return 0;
  }
  if (!ss_buffer_append(buffer, parser->input + parser->position, width)) {
    ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
    return 0;
  }
  parser->position += width;
  return 1;
}

static int ss_parse_string(SsParser *parser, SsString *result) {
  SsBuffer buffer = {0};
  if (parser->position >= parser->size ||
      parser->input[parser->position] != (unsigned char)'"') {
    ss_set_error(parser, SS_JSON_INVALID_JSON);
    return 0;
  }
  ++parser->position;
  while (parser->position < parser->size) {
    unsigned char byte = parser->input[parser->position];
    if (byte == (unsigned char)'"') {
      ++parser->position;
      result->data = buffer.data;
      result->size = buffer.size;
      return 1;
    }
    if (byte < 0x20U) {
      ss_set_error(parser, SS_JSON_INVALID_JSON);
      free(buffer.data);
      return 0;
    }
    if (byte == (unsigned char)'\\') {
      uint32_t codepoint;
      ++parser->position;
      if (parser->position >= parser->size) {
        ss_set_error(parser, SS_JSON_INVALID_JSON);
        free(buffer.data);
        return 0;
      }
      byte = parser->input[parser->position++];
      switch (byte) {
      case (unsigned char)'"':
      case (unsigned char)'\\':
      case (unsigned char)'/':
        if (!ss_buffer_byte(&buffer, byte)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      case (unsigned char)'b':
        if (!ss_buffer_byte(&buffer, 0x08U)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      case (unsigned char)'f':
        if (!ss_buffer_byte(&buffer, 0x0cU)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      case (unsigned char)'n':
        if (!ss_buffer_byte(&buffer, 0x0aU)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      case (unsigned char)'r':
        if (!ss_buffer_byte(&buffer, 0x0dU)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      case (unsigned char)'t':
        if (!ss_buffer_byte(&buffer, 0x09U)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      case (unsigned char)'u':
        if (!ss_parse_hex4(parser, &codepoint)) {
          free(buffer.data);
          return 0;
        }
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
          uint32_t low;
          if (parser->size - parser->position < 6U ||
              parser->input[parser->position] != (unsigned char)'\\' ||
              parser->input[parser->position + 1U] != (unsigned char)'u') {
            ss_set_error(parser, SS_JSON_INVALID_UNICODE);
            free(buffer.data);
            return 0;
          }
          parser->position += 2U;
          if (!ss_parse_hex4(parser, &low)) {
            free(buffer.data);
            return 0;
          }
          if (low < 0xdc00U || low > 0xdfffU) {
            ss_set_error(parser, SS_JSON_INVALID_UNICODE);
            free(buffer.data);
            return 0;
          }
          codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
          ss_set_error(parser, SS_JSON_INVALID_UNICODE);
          free(buffer.data);
          return 0;
        }
        if (!ss_append_codepoint(&buffer, codepoint)) {
          ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
          free(buffer.data);
          return 0;
        }
        break;
      default:
        ss_set_error(parser, SS_JSON_INVALID_JSON);
        free(buffer.data);
        return 0;
      }
    } else if (!ss_copy_utf8(parser, &buffer)) {
      free(buffer.data);
      return 0;
    }
  }
  ss_set_error(parser, SS_JSON_INVALID_JSON);
  free(buffer.data);
  return 0;
}

static int ss_string_equal(const SsString *left, const SsString *right) {
  return left->size == right->size &&
         (left->size == 0U || memcmp(left->data, right->data, left->size) == 0);
}

static int ss_double_equal(double left, double right) {
  uint64_t left_bits;
  uint64_t right_bits;
  memcpy(&left_bits, &left, sizeof(left_bits));
  memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

static int ss_normalize_exponent(char *text, size_t capacity) {
  char *exponent = strchr(text, 'e');
  char *end;
  long value;
  int written;
  char mantissa[64];
  size_t mantissa_size;
  if (exponent == NULL) {
    exponent = strchr(text, 'E');
  }
  if (exponent == NULL) {
    return 1;
  }
  mantissa_size = (size_t)(exponent - text);
  if (mantissa_size >= sizeof(mantissa)) {
    return 0;
  }
  memcpy(mantissa, text, mantissa_size);
  mantissa[mantissa_size] = '\0';
  errno = 0;
  value = strtol(exponent + 1, &end, 10);
  if (errno != 0 || *end != '\0') {
    return 0;
  }
  written = snprintf(text, capacity, "%se%ld", mantissa, value);
  return written >= 0 && (size_t)written < capacity;
}

static int ss_format_double(double value, SsString *result) {
  int precision;
  char candidate[128];
  char *end;
  for (precision = 1; precision <= 17; ++precision) {
    int written = snprintf(candidate, sizeof(candidate), "%.*g", precision, value);
    double parsed;
    if (written < 0 || (size_t)written >= sizeof(candidate)) {
      return 0;
    }
    if (!ss_normalize_exponent(candidate, sizeof(candidate))) {
      return 0;
    }
    errno = 0;
    parsed = strtod(candidate, &end);
    if (errno == 0 && *end == '\0' && ss_double_equal(parsed, value)) {
      size_t size = strlen(candidate);
      result->data = (unsigned char *)malloc(size);
      if (result->data == NULL && size != 0U) {
        return 0;
      }
      memcpy(result->data, candidate, size);
      result->size = size;
      return 1;
    }
  }
  return 0;
}

static SsValue *ss_parse_value(SsParser *parser);

static SsValue *ss_parse_number(SsParser *parser) {
  size_t start = parser->position;
  int has_fraction_or_exponent = 0;
  SsValue *value;
  if (parser->input[parser->position] == (unsigned char)'-') {
    ++parser->position;
  }
  if (parser->position >= parser->size) {
    ss_set_error(parser, SS_JSON_INVALID_NUMBER);
    return NULL;
  }
  if (parser->input[parser->position] == (unsigned char)'0') {
    ++parser->position;
    if (parser->position < parser->size &&
        isdigit((int)parser->input[parser->position]) != 0) {
      ss_set_error(parser, SS_JSON_INVALID_NUMBER);
      return NULL;
    }
  } else if (parser->input[parser->position] >= (unsigned char)'1' &&
             parser->input[parser->position] <= (unsigned char)'9') {
    while (parser->position < parser->size &&
           isdigit((int)parser->input[parser->position]) != 0) {
      ++parser->position;
    }
  } else {
    ss_set_error(parser, SS_JSON_INVALID_NUMBER);
    return NULL;
  }
  if (parser->position < parser->size &&
      parser->input[parser->position] == (unsigned char)'.') {
    has_fraction_or_exponent = 1;
    ++parser->position;
    if (parser->position >= parser->size ||
        isdigit((int)parser->input[parser->position]) == 0) {
      ss_set_error(parser, SS_JSON_INVALID_NUMBER);
      return NULL;
    }
    while (parser->position < parser->size &&
           isdigit((int)parser->input[parser->position]) != 0) {
      ++parser->position;
    }
  }
  if (parser->position < parser->size &&
      (parser->input[parser->position] == (unsigned char)'e' ||
       parser->input[parser->position] == (unsigned char)'E')) {
    has_fraction_or_exponent = 1;
    ++parser->position;
    if (parser->position < parser->size &&
        (parser->input[parser->position] == (unsigned char)'+' ||
         parser->input[parser->position] == (unsigned char)'-')) {
      ++parser->position;
    }
    if (parser->position >= parser->size ||
        isdigit((int)parser->input[parser->position]) == 0) {
      ss_set_error(parser, SS_JSON_INVALID_NUMBER);
      return NULL;
    }
    while (parser->position < parser->size &&
           isdigit((int)parser->input[parser->position]) != 0) {
      ++parser->position;
    }
  }
  value = ss_value_create(parser, SS_VALUE_NUMBER);
  if (value == NULL) {
    return NULL;
  }
  if (!has_fraction_or_exponent) {
    size_t size = parser->position - start;
    if (size == 2U && parser->input[start] == (unsigned char)'-' &&
        parser->input[start + 1U] == (unsigned char)'0') {
      ss_set_error(parser, SS_JSON_AMBIGUOUS_NUMBER);
      ss_value_destroy(value);
      return NULL;
    }
    value->as.number.data = (unsigned char *)malloc(size);
    if (value->as.number.data == NULL && size != 0U) {
      ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
      ss_value_destroy(value);
      return NULL;
    }
    memcpy(value->as.number.data, parser->input + start, size);
    value->as.number.size = size;
  } else {
    size_t size = parser->position - start;
    char *lexeme = (char *)malloc(size + 1U);
    char *end;
    double parsed;
    if (lexeme == NULL) {
      ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
      ss_value_destroy(value);
      return NULL;
    }
    memcpy(lexeme, parser->input + start, size);
    lexeme[size] = '\0';
    errno = 0;
    parsed = strtod(lexeme, &end);
    if (errno != 0 || *end != '\0' || !isfinite(parsed)) {
      free(lexeme);
      ss_set_error(parser, SS_JSON_INVALID_NUMBER);
      ss_value_destroy(value);
      return NULL;
    }
    if (parsed == 0.0 && signbit(parsed) != 0) {
      free(lexeme);
      ss_set_error(parser, SS_JSON_AMBIGUOUS_NUMBER);
      ss_value_destroy(value);
      return NULL;
    }
    free(lexeme);
    if (!ss_format_double(parsed, &value->as.number)) {
      ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
      ss_value_destroy(value);
      return NULL;
    }
  }
  return value;
}

static int ss_match_literal(SsParser *parser, const char *literal) {
  size_t size = strlen(literal);
  if (parser->size - parser->position < size ||
      memcmp(parser->input + parser->position, literal, size) != 0) {
    return 0;
  }
  parser->position += size;
  return 1;
}

static SsValue *ss_parse_array(SsParser *parser) {
  SsValue *value = ss_value_create(parser, SS_VALUE_ARRAY);
  if (value == NULL) {
    return NULL;
  }
  ++parser->position;
  ss_skip_space(parser);
  if (parser->position < parser->size &&
      parser->input[parser->position] == (unsigned char)']') {
    ++parser->position;
    return value;
  }
  while (parser->position < parser->size) {
    SsValue *item;
    ss_skip_space(parser);
    item = ss_parse_value(parser);
    if (item == NULL) {
      ss_value_destroy(value);
      return NULL;
    }
    if (value->as.array.count == value->as.array.capacity) {
      size_t capacity =
          value->as.array.capacity == 0U ? 16U : value->as.array.capacity * 2U;
      SsValue **items;
      if (capacity < value->as.array.capacity ||
          capacity > SIZE_MAX / sizeof(*value->as.array.items)) {
        ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
        ss_value_destroy(item);
        ss_value_destroy(value);
        return NULL;
      }
      items = (SsValue **)realloc(value->as.array.items,
                                  capacity * sizeof(*value->as.array.items));
      if (items == NULL) {
        ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
        ss_value_destroy(item);
        ss_value_destroy(value);
        return NULL;
      }
      value->as.array.items = items;
      value->as.array.capacity = capacity;
    }
    value->as.array.items[value->as.array.count++] = item;
    ss_skip_space(parser);
    if (parser->position >= parser->size) {
      break;
    }
    if (parser->input[parser->position] == (unsigned char)']') {
      ++parser->position;
      return value;
    }
    if (parser->input[parser->position] != (unsigned char)',') {
      break;
    }
    ++parser->position;
  }
  ss_set_error(parser, SS_JSON_INVALID_JSON);
  ss_value_destroy(value);
  return NULL;
}

static SsValue *ss_parse_object(SsParser *parser) {
  SsValue *value = ss_value_create(parser, SS_VALUE_OBJECT);
  if (value == NULL) {
    return NULL;
  }
  ++parser->position;
  ss_skip_space(parser);
  if (parser->position < parser->size &&
      parser->input[parser->position] == (unsigned char)'}') {
    ++parser->position;
    return value;
  }
  while (parser->position < parser->size) {
    SsString key = {0};
    SsValue *member_value;
    SsMember *members;
    size_t index;
    ss_skip_space(parser);
    if (!ss_parse_string(parser, &key)) {
      ss_value_destroy(value);
      return NULL;
    }
    for (index = 0U; index < value->as.object.count; ++index) {
      if (ss_string_equal(&key, &value->as.object.members[index].key)) {
        ss_string_destroy(&key);
        ss_set_error(parser, SS_JSON_DUPLICATE_KEY);
        ss_value_destroy(value);
        return NULL;
      }
    }
    ss_skip_space(parser);
    if (parser->position >= parser->size ||
        parser->input[parser->position] != (unsigned char)':') {
      ss_string_destroy(&key);
      ss_set_error(parser, SS_JSON_INVALID_JSON);
      ss_value_destroy(value);
      return NULL;
    }
    ++parser->position;
    ss_skip_space(parser);
    member_value = ss_parse_value(parser);
    if (member_value == NULL) {
      ss_string_destroy(&key);
      ss_value_destroy(value);
      return NULL;
    }
    members = (SsMember *)realloc(value->as.object.members,
                                  (value->as.object.count + 1U) *
                                      sizeof(*value->as.object.members));
    if (members == NULL) {
      ss_set_error(parser, SS_JSON_OUT_OF_MEMORY);
      ss_string_destroy(&key);
      ss_value_destroy(member_value);
      ss_value_destroy(value);
      return NULL;
    }
    value->as.object.members = members;
    value->as.object.members[value->as.object.count].key = key;
    value->as.object.members[value->as.object.count].value = member_value;
    ++value->as.object.count;
    ss_skip_space(parser);
    if (parser->position >= parser->size) {
      break;
    }
    if (parser->input[parser->position] == (unsigned char)'}') {
      ++parser->position;
      return value;
    }
    if (parser->input[parser->position] != (unsigned char)',') {
      break;
    }
    ++parser->position;
  }
  ss_set_error(parser, SS_JSON_INVALID_JSON);
  ss_value_destroy(value);
  return NULL;
}

static SsValue *ss_parse_value(SsParser *parser) {
  SsValue *value;
  ss_skip_space(parser);
  if (parser->position >= parser->size) {
    ss_set_error(parser, SS_JSON_INVALID_JSON);
    return NULL;
  }
  switch (parser->input[parser->position]) {
  case (unsigned char)'n':
    if (!ss_match_literal(parser, "null")) {
      ss_set_error(parser, SS_JSON_INVALID_JSON);
      return NULL;
    }
    return ss_value_create(parser, SS_VALUE_NULL);
  case (unsigned char)'t':
    if (!ss_match_literal(parser, "true")) {
      ss_set_error(parser, SS_JSON_INVALID_JSON);
      return NULL;
    }
    value = ss_value_create(parser, SS_VALUE_BOOL);
    if (value != NULL) {
      value->as.boolean = 1;
    }
    return value;
  case (unsigned char)'f':
    if (!ss_match_literal(parser, "false")) {
      ss_set_error(parser, SS_JSON_INVALID_JSON);
      return NULL;
    }
    value = ss_value_create(parser, SS_VALUE_BOOL);
    if (value != NULL) {
      value->as.boolean = 0;
    }
    return value;
  case (unsigned char)'"':
    value = ss_value_create(parser, SS_VALUE_STRING);
    if (value == NULL) {
      return NULL;
    }
    if (!ss_parse_string(parser, &value->as.string)) {
      ss_value_destroy(value);
      return NULL;
    }
    return value;
  case (unsigned char)'[':
    return ss_parse_array(parser);
  case (unsigned char)'{':
    return ss_parse_object(parser);
  case (unsigned char)'-':
  case (unsigned char)'0':
  case (unsigned char)'1':
  case (unsigned char)'2':
  case (unsigned char)'3':
  case (unsigned char)'4':
  case (unsigned char)'5':
  case (unsigned char)'6':
  case (unsigned char)'7':
  case (unsigned char)'8':
  case (unsigned char)'9':
    return ss_parse_number(parser);
  case (unsigned char)'N':
  case (unsigned char)'I':
    ss_set_error(parser, SS_JSON_INVALID_NUMBER);
    return NULL;
  default:
    ss_set_error(parser, SS_JSON_INVALID_JSON);
    return NULL;
  }
}

static int ss_member_compare(const void *left_pointer, const void *right_pointer) {
  const SsMember *left = (const SsMember *)left_pointer;
  const SsMember *right = (const SsMember *)right_pointer;
  size_t shared = left->key.size < right->key.size ? left->key.size : right->key.size;
  int comparison = shared == 0U ? 0 : memcmp(left->key.data, right->key.data, shared);
  if (comparison != 0) {
    return comparison;
  }
  if (left->key.size < right->key.size) {
    return -1;
  }
  if (left->key.size > right->key.size) {
    return 1;
  }
  return 0;
}

static int ss_encode_string(const SsString *string, SsBuffer *output) {
  static const char hexadecimal[] = "0123456789abcdef";
  size_t index;
  if (!ss_buffer_byte(output, (unsigned char)'"')) {
    return 0;
  }
  for (index = 0U; index < string->size; ++index) {
    unsigned char byte = string->data[index];
    const char *escape = NULL;
    switch (byte) {
    case (unsigned char)'"':
      escape = "\\\"";
      break;
    case (unsigned char)'\\':
      escape = "\\\\";
      break;
    case 0x08U:
      escape = "\\b";
      break;
    case 0x0cU:
      escape = "\\f";
      break;
    case 0x0aU:
      escape = "\\n";
      break;
    case 0x0dU:
      escape = "\\r";
      break;
    case 0x09U:
      escape = "\\t";
      break;
    default:
      break;
    }
    if (escape != NULL) {
      if (!ss_buffer_append(output, escape, 2U)) {
        return 0;
      }
    } else if (byte < 0x20U) {
      unsigned char encoded[6] = {(unsigned char)'\\',
                                  (unsigned char)'u',
                                  (unsigned char)'0',
                                  (unsigned char)'0',
                                  (unsigned char)hexadecimal[byte >> 4U],
                                  (unsigned char)hexadecimal[byte & 0x0fU]};
      if (!ss_buffer_append(output, encoded, sizeof(encoded))) {
        return 0;
      }
    } else if (!ss_buffer_byte(output, byte)) {
      return 0;
    }
  }
  return ss_buffer_byte(output, (unsigned char)'"');
}

static int ss_encode_value(SsValue *value, SsBuffer *output) {
  size_t index;
  switch (value->type) {
  case SS_VALUE_NULL:
    return ss_buffer_append(output, "null", 4U);
  case SS_VALUE_BOOL:
    return value->as.boolean != 0 ? ss_buffer_append(output, "true", 4U)
                                  : ss_buffer_append(output, "false", 5U);
  case SS_VALUE_NUMBER:
    return ss_buffer_append(output, value->as.number.data, value->as.number.size);
  case SS_VALUE_STRING:
    return ss_encode_string(&value->as.string, output);
  case SS_VALUE_ARRAY:
    if (!ss_buffer_byte(output, (unsigned char)'[')) {
      return 0;
    }
    for (index = 0U; index < value->as.array.count; ++index) {
      if (index != 0U && !ss_buffer_byte(output, (unsigned char)',')) {
        return 0;
      }
      if (!ss_encode_value(value->as.array.items[index], output)) {
        return 0;
      }
    }
    return ss_buffer_byte(output, (unsigned char)']');
  case SS_VALUE_OBJECT:
    ss_sort(value->as.object.members, value->as.object.count,
            sizeof(*value->as.object.members), ss_member_compare);
    if (!ss_buffer_byte(output, (unsigned char)'{')) {
      return 0;
    }
    for (index = 0U; index < value->as.object.count; ++index) {
      if (index != 0U && !ss_buffer_byte(output, (unsigned char)',')) {
        return 0;
      }
      if (!ss_encode_string(&value->as.object.members[index].key, output) ||
          !ss_buffer_byte(output, (unsigned char)':') ||
          !ss_encode_value(value->as.object.members[index].value, output)) {
        return 0;
      }
    }
    return ss_buffer_byte(output, (unsigned char)'}');
  }
  return 0;
}

SsJsonStatus ss_json_canonicalize(const unsigned char *input, size_t input_size,
                                  unsigned char **output, size_t *output_size,
                                  SsJsonError *error) {
  SsParser parser = {0};
  SsValue *value;
  SsBuffer buffer = {0};
  if (output == NULL || output_size == NULL || error == NULL ||
      (input == NULL && input_size != 0U)) {
    return SS_JSON_INVALID_JSON;
  }
  *output = NULL;
  *output_size = 0U;
  parser.input = input;
  parser.size = input_size;
  parser.error.status = SS_JSON_OK;
  value = ss_parse_value(&parser);
  if (value != NULL) {
    ss_skip_space(&parser);
    if (parser.position != parser.size) {
      ss_set_error(&parser, SS_JSON_TRAILING_DATA);
    }
  }
  if (parser.error.status == SS_JSON_OK && !ss_encode_value(value, &buffer)) {
    ss_set_error(&parser, SS_JSON_OUT_OF_MEMORY);
  }
  ss_value_destroy(value);
  if (parser.error.status != SS_JSON_OK) {
    free(buffer.data);
    *error = parser.error;
    return parser.error.status;
  }
  *output = buffer.data;
  *output_size = buffer.size;
  error->status = SS_JSON_OK;
  error->offset = 0U;
  return SS_JSON_OK;
}

const char *ss_json_status_name(SsJsonStatus status) {
  switch (status) {
  case SS_JSON_OK:
    return "ok";
  case SS_JSON_INVALID_JSON:
    return "invalid_json";
  case SS_JSON_DUPLICATE_KEY:
    return "duplicate_key";
  case SS_JSON_INVALID_NUMBER:
    return "invalid_number";
  case SS_JSON_AMBIGUOUS_NUMBER:
    return "ambiguous_number";
  case SS_JSON_INVALID_UNICODE:
    return "invalid_unicode";
  case SS_JSON_TRAILING_DATA:
    return "trailing_data";
  case SS_JSON_OUT_OF_MEMORY:
    return "out_of_memory";
  }
  return "unknown";
}

void ss_json_free(void *pointer) { free(pointer); }
