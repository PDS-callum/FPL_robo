# FPL Backtest Report

**Test Season:** 2023-24
**Target Model:** points_scored
**Budget:** £100.0m
**Test Date:** 2025-08-22

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| Total Points | 0 |
| Net Points (after transfers) | 0 |
| Total Transfers | 0 |
| Transfer Cost | 0 |
| Gameweeks Simulated | 0 |
| Avg Points/GW | 0.00 |
| Avg Net Points/GW | 0.00 |
| Best Gameweek | 0 points |
| Worst Gameweek | 0 points |
| Avg Transfers/GW | 0.00 |

## 🏋️ Training Details

**Training Seasons:** 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23

## 🤖 Model Information

- **Architecture:** Deep Neural Network
- **Target Variable:** points_scored
- **Feature Engineering:** Rolling averages, position encoding, form metrics
- **Optimization:** Greedy selection with budget and formation constraints

## 📝 Notes

- This backtest simulates selecting an optimal team each gameweek based on model predictions
- Transfer costs are calculated assuming 1 free transfer per gameweek
- Team selection uses a greedy algorithm optimized for predicted points per cost
- Formation constraints and budget limits are enforced
