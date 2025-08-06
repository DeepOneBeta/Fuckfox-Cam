#ifndef __I2C_CLIENT_H
#define __I2C_CLIENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"  

#define CMD_GET_TEMP    0x01
#define CMD_GET_HUMI    0x02
#define CMD_GET_PRESS   0x03
#define CMD_GET_ALL     0x04
#define CMD_SET_LED     0x10
#define CMD_RESET       0xFF


extern uint8_t temp;
extern uint8_t humi;
extern uint8_t press;

#define RxData_Long       10
extern uint8_t RxData[RxData_Long];         // 接收缓冲区（volatile 更佳）
extern uint8_t RxData_save[RxData_Long];    // 数据保存缓冲区
extern uint8_t status[16];        // 状态信息（可选扩展）

extern uint8_t tx_csv_buffer[20];    // 发送给主设备的 CSV 数据缓冲区

HAL_StatusTypeDef BuildResponseFromCommands(void);

#ifdef __cplusplus
}
#endif

#endif 


