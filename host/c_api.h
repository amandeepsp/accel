#ifndef ACCEL_C_API_H
#define ACCEL_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AccelHandle AccelHandle;

typedef enum {
  ACCEL_OK = 0,
  ACCEL_INVALID_ARGUMENT = 1,
  ACCEL_PAYLOAD_TOO_LARGE = 2,
  ACCEL_BAD_RESPONSE = 3,
  ACCEL_BAD_MAGIC = 4,
  ACCEL_UNKNOWN_OP = 5,
  ACCEL_BAD_PAYLOAD_LEN = 6,
  ACCEL_BAD_ADDRESS = 7,
  ACCEL_ILLEGAL_INSTRUCTION = 8,
  ACCEL_TRAP_FAULT = 9,
  ACCEL_DEVICE_ERROR = 10,
  ACCEL_OUT_OF_MEMORY = 11,
  ACCEL_IO_ERROR = 12,
} accel_status_t;

accel_status_t accel_open(const char *port_path, uint32_t baud_rate, AccelHandle **out_handle);
void accel_close(AccelHandle *handle);
accel_status_t accel_ping(AccelHandle *handle);
uint16_t accel_last_cycles(AccelHandle *handle);
const char *accel_status_string(accel_status_t code);

accel_status_t accel_write_mem(AccelHandle *handle, uint32_t addr, const uint8_t *data, size_t len);
accel_status_t accel_read_mem(AccelHandle *handle, uint32_t addr, uint8_t *buf, size_t len);
accel_status_t accel_exec(AccelHandle *handle, const uint8_t *program, size_t program_len, uint32_t *out_cycles);

#ifdef __cplusplus
}
#endif

#endif
