# Efficient Data Stream Anomaly Detection

## Description

This project implements an anomaly detection system using a voting ensemble approach. It combines multiple anomaly detection algorithms, such as Exponential Moving Average (EMA), Seasonal Hybrid Extreme Studentized Deviate (SHESD), Quantile-Based Detection, and Random Cut Trees (RCTrees), to effectively identify anomalies in a data stream.

## Features

- Generates a sinusoidal data stream with added anomalies. (If you want to use your data read you data from csv and into an array and replace data stream with your personal data).
- Implements various anomaly detection algorithms.
- Visualizes the data stream along with detected anomalies.
- You can change the weights you want to provide to model based on your data i by editing the "detectors" dictionary.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shalin06/AnomalyDetection.git
   cd AnomalyDetection
   pip install -r requirements.txt
   ```
