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
extern uint8_t RxData[RxData_Long];         // ���ջ�������volatile ���ѣ�
extern uint8_t RxData_save[RxData_Long];    // ���ݱ��滺����
extern uint8_t status[16];        // ״̬��Ϣ����ѡ��չ��

extern uint8_t tx_csv_buffer[20];    // ���͸����豸�� CSV ���ݻ�����

HAL_StatusTypeDef BuildResponseFromCommands(void);

#ifdef __cplusplus
}
#endif

#endif 


