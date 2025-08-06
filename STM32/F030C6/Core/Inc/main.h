/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f0xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define DIAG_F_Pin GPIO_PIN_13
#define DIAG_F_GPIO_Port GPIOC
#define DIAG_X_Pin GPIO_PIN_14
#define DIAG_X_GPIO_Port GPIOC
#define DIAG_C_Pin GPIO_PIN_15
#define DIAG_C_GPIO_Port GPIOC
#define MS2_F_Pin GPIO_PIN_4
#define MS2_F_GPIO_Port GPIOA
#define MS1_F_Pin GPIO_PIN_5
#define MS1_F_GPIO_Port GPIOA
#define DIR_F_Pin GPIO_PIN_6
#define DIR_F_GPIO_Port GPIOA
#define STEP_F_Pin GPIO_PIN_7
#define STEP_F_GPIO_Port GPIOA
#define FAN_Pin GPIO_PIN_0
#define FAN_GPIO_Port GPIOB
#define STEP_C_Pin GPIO_PIN_1
#define STEP_C_GPIO_Port GPIOB
#define DIR_C_Pin GPIO_PIN_2
#define DIR_C_GPIO_Port GPIOB
#define MS1_C_Pin GPIO_PIN_10
#define MS1_C_GPIO_Port GPIOB
#define MS2_C_Pin GPIO_PIN_11
#define MS2_C_GPIO_Port GPIOB
#define BRB_C_Pin GPIO_PIN_12
#define BRB_C_GPIO_Port GPIOB
#define BRA_C_Pin GPIO_PIN_13
#define BRA_C_GPIO_Port GPIOB
#define BRB_X_Pin GPIO_PIN_14
#define BRB_X_GPIO_Port GPIOB
#define BRA_X_Pin GPIO_PIN_15
#define BRA_X_GPIO_Port GPIOB
#define F_STOP_Pin GPIO_PIN_8
#define F_STOP_GPIO_Port GPIOA
#define DACOUT_Pin GPIO_PIN_9
#define DACOUT_GPIO_Port GPIOA
#define BRB_F_Pin GPIO_PIN_10
#define BRB_F_GPIO_Port GPIOA
#define BRA_F_Pin GPIO_PIN_11
#define BRA_F_GPIO_Port GPIOA
#define MS_X_Pin GPIO_PIN_15
#define MS_X_GPIO_Port GPIOA
#define DIR_X_Pin GPIO_PIN_3
#define DIR_X_GPIO_Port GPIOB
#define STEP_X_Pin GPIO_PIN_4
#define STEP_X_GPIO_Port GPIOB
#define MS1_X_Pin GPIO_PIN_5
#define MS1_X_GPIO_Port GPIOB
#define SEL0_Pin GPIO_PIN_6
#define SEL0_GPIO_Port GPIOB
#define SEL1_Pin GPIO_PIN_7
#define SEL1_GPIO_Port GPIOB
#define SEL2_Pin GPIO_PIN_8
#define SEL2_GPIO_Port GPIOB
#define VREF_SW_Pin GPIO_PIN_9
#define VREF_SW_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
