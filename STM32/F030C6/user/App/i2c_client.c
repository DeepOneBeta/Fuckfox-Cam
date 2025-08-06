#include "i2c_client.h"
#include "i2c.h"
#include <string.h>
#include <stdio.h>

uint8_t temp = 25;
uint8_t humi = 60;
uint8_t press = 103;

uint8_t RxData[RxData_Long];         // ���ջ�������volatile ���ѣ�
uint8_t RxData_save[RxData_Long];    // ���ݱ��滺����

uint8_t status[16];        // ״̬��Ϣ����ѡ��չ��
uint8_t tx_csv_buffer[20];    // ���͸����豸�� CSV ���ݻ�����

HAL_StatusTypeDef BuildResponseFromCommands(void) {
    // ��ջ�����
    memset(tx_csv_buffer, 0, sizeof(tx_csv_buffer));

    // ֻ�����һ�������ֽڣ��㲻��Ҫѭ�� 21 �Σ�
    uint8_t cmd = RxData_save[0];

    switch (cmd) {
        case CMD_GET_TEMP:
            sprintf((char*)tx_csv_buffer, "%d", temp);
            break;

        case CMD_GET_HUMI:
            sprintf((char*)tx_csv_buffer, "%d", humi);
            break;

        case CMD_GET_PRESS:
            sprintf((char*)tx_csv_buffer, "%d", press);
            break;

        case CMD_GET_ALL:
            // ���� "temp,humi,press"
            sprintf((char*)tx_csv_buffer, "%d,%d,%d", temp, humi, press);
            break;

        default:
            // ��Ч����
            strcpy((char*)tx_csv_buffer, "ERR");
            return HAL_ERROR;
    }

    return HAL_OK;
}

// ----------------- HAL�⹳�Ӻ��� -----------------
void HAL_I2C_AddrCallback(I2C_HandleTypeDef *hi2c, uint8_t TransferDirection, uint16_t AddrMatchCode)
{
  if (hi2c->Instance == I2C1) {
		if (TransferDirection == I2C_DIRECTION_TRANSMIT) {
			HAL_I2C_Slave_Seq_Receive_DMA(hi2c, RxData, sizeof(RxData),I2C_FIRST_AND_LAST_FRAME);
		}
		else {
			HAL_I2C_Slave_Seq_Transmit_DMA(hi2c, tx_csv_buffer, sizeof(tx_csv_buffer), I2C_FIRST_AND_LAST_FRAME);
		}
	}
}

 void HAL_I2C_SlaveRxCpltCallback(I2C_HandleTypeDef *hi2c) {
    if (hi2c->Instance == I2C1) {
        for (int i = 0; i < (RxData_Long+1); i++) {
            RxData_save[i] = RxData[i];
            RxData[i] = 0x00;
        }
        if(BuildResponseFromCommands() == HAL_OK) {
            HAL_I2C_Slave_Seq_Transmit_DMA(hi2c, tx_csv_buffer, sizeof(tx_csv_buffer), I2C_FIRST_AND_LAST_FRAME);
        }
	}
 }

void HAL_I2C_SlaveTxCpltCallback(I2C_HandleTypeDef *hi2c) {
    if (hi2c->Instance == I2C1) {

	  }
}

void HAL_I2C_ListenCpltCallback(I2C_HandleTypeDef *hi2c) {
	  if (hi2c->Instance == I2C1) {

	  }
    HAL_I2C_EnableListen_IT(hi2c);
    HAL_I2C_Slave_Seq_Receive_DMA(hi2c, RxData, sizeof(RxData),I2C_FIRST_AND_LAST_FRAME);
}

void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c) {
    if (hi2c->Instance == I2C1) {
        // �������������
        if (hi2c->ErrorCode & HAL_I2C_ERROR_BERR) {
            // �������ߴ���
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_BERR);
        }
        if (hi2c->ErrorCode & HAL_I2C_ERROR_ARLO) {
            // �����ٲö�ʧ������������
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_ARLO);
        }
       if (hi2c->ErrorCode & HAL_I2C_ERROR_AF) {
            // ? ��ܿ���������������
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_AF);  // ��� AF ��־

            // �ֶ�����������ɡ��߼�
            uint16_t received = sizeof(RxData) - hi2c->XferCount;
            if (received > 0) {
                for (int i = 0; i < received; i++) {
                    RxData_save[i] = RxData[i];
                    RxData[i] = 0x00;
                }

                if (BuildResponseFromCommands() == HAL_OK) {
                    HAL_I2C_Slave_Seq_Transmit_DMA(hi2c, tx_csv_buffer, strlen((char*)tx_csv_buffer), I2C_FIRST_AND_LAST_FRAME);
                } else {
                    // ������������
                    HAL_I2C_Slave_Seq_Receive_DMA(hi2c, RxData, sizeof(RxData), I2C_FIRST_AND_LAST_FRAME);
                }
            } else {
                // ��ĳ����ˣ���������
                HAL_I2C_Slave_Seq_Receive_DMA(hi2c, RxData, sizeof(RxData), I2C_FIRST_AND_LAST_FRAME);
            }

            return; // ������ AF��ֱ�ӷ���
        }
        if (hi2c->ErrorCode & HAL_I2C_ERROR_OVR) {
            // �������
            // �����ڣ����ط�̫�죬�ӻ�û������
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_OVR);
        }
        if (hi2c->ErrorCode & HAL_I2C_ERROR_DMA) {
            // DMA �������
        }
        __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_BERR | I2C_FLAG_ARLO | I2C_FLAG_OVR);
    }
}
