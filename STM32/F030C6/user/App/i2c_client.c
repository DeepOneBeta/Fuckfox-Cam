#include "i2c_client.h"
#include "i2c.h"
#include <string.h>
#include <stdio.h>

uint8_t temp = 25;
uint8_t humi = 60;
uint8_t press = 103;

uint8_t RxData[RxData_Long];         // 接收缓冲区（volatile 更佳）
uint8_t RxData_save[RxData_Long];    // 数据保存缓冲区

uint8_t status[16];        // 状态信息（可选扩展）
uint8_t tx_csv_buffer[20];    // 发送给主设备的 CSV 数据缓冲区

HAL_StatusTypeDef BuildResponseFromCommands(void) {
    // 清空缓冲区
    memset(tx_csv_buffer, 0, sizeof(tx_csv_buffer));

    // 只处理第一个命令字节（你不需要循环 21 次）
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
            // 返回 "temp,humi,press"
            sprintf((char*)tx_csv_buffer, "%d,%d,%d", temp, humi, press);
            break;

        default:
            // 无效命令
            strcpy((char*)tx_csv_buffer, "ERR");
            return HAL_ERROR;
    }

    return HAL_OK;
}

// ----------------- HAL库钩子函数 -----------------
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
        // 检查具体错误类型
        if (hi2c->ErrorCode & HAL_I2C_ERROR_BERR) {
            // 处理总线错误
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_BERR);
        }
        if (hi2c->ErrorCode & HAL_I2C_ERROR_ARLO) {
            // 处理仲裁丢失（多主竞争）
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_ARLO);
        }
       if (hi2c->ErrorCode & HAL_I2C_ERROR_AF) {
            // ? 这很可能是正常结束！
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_AF);  // 清除 AF 标志

            // 手动处理“接收完成”逻辑
            uint16_t received = sizeof(RxData) - hi2c->XferCount;
            if (received > 0) {
                for (int i = 0; i < received; i++) {
                    RxData_save[i] = RxData[i];
                    RxData[i] = 0x00;
                }

                if (BuildResponseFromCommands() == HAL_OK) {
                    HAL_I2C_Slave_Seq_Transmit_DMA(hi2c, tx_csv_buffer, strlen((char*)tx_csv_buffer), I2C_FIRST_AND_LAST_FRAME);
                } else {
                    // 重新启动接收
                    HAL_I2C_Slave_Seq_Receive_DMA(hi2c, RxData, sizeof(RxData), I2C_FIRST_AND_LAST_FRAME);
                }
            } else {
                // 真的出错了，重启接收
                HAL_I2C_Slave_Seq_Receive_DMA(hi2c, RxData, sizeof(RxData), I2C_FIRST_AND_LAST_FRAME);
            }

            return; // 处理完 AF，直接返回
        }
        if (hi2c->ErrorCode & HAL_I2C_ERROR_OVR) {
            // 处理溢出
            // 常见于：主控发太快，从机没处理完
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_OVR);
        }
        if (hi2c->ErrorCode & HAL_I2C_ERROR_DMA) {
            // DMA 传输错误
        }
        __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_BERR | I2C_FLAG_ARLO | I2C_FLAG_OVR);
    }
}
