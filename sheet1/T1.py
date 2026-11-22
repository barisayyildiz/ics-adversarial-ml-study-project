#!/usr/bin/env python
# coding: utf-8

# # Study Project: Adversarial Machine Learning
# ## Part I: Statistical Analysis and Generation of Process Data
# ## Group 15
# ## SIN Wang Yat
# 
# ### Task 1: Statistical Analysis of Physical Readings
# 
# The goal of this task is to analyse the properties of the WADI datasets, WADI_14days and WADI_attackdata. 
# 
# - Compute K-S statistic and compare the sensor readings from datasets
# - Count the set of system states and the percentages
# 

# ### Libray Import and Data Loading

print(f"Start: Import Library and Load the datasets")
# Library Import
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

if __name__ == '__main__':
	# Define the base directory path
	base_path = Path(__file__).resolve().parent.parent / "data/sheet1/"   # sheet 3: was ~/Projects/P1/data/
	cache_train_file = base_path / "processed/cache_train_file.parquet"
	cache_test_file = base_path / "processed/cache_test_file.parquet"

	# Load normal 14days data and attack data from CSV or Cache
	# Load 14days Dataset
	if cache_train_file.exists():
	   print("Loading fast cached Parquet data-14days dataset")
	   train17_df = pd.read_parquet(cache_train_file)
	else:
	   print("Cache not found. Load data from CSV file-14days dataset")
	   # Skip 4 meta data rows 
	   train17_df = pd.read_csv(base_path / "raw/2017/WADI_14days.csv", skiprows=4)
	   # Save the cache for next time
	   train17_df.to_parquet(cache_train_file)

	# Load attackdata Dataset
	if cache_test_file.exists():
	   print("Loading fast cached Parquet data-attackdata dataset")
	   test17_df = pd.read_parquet(cache_test_file)
	else:
	   print("Cache not found. Load data from CSV file-attackdata dataset")
	   test17_df = pd.read_csv(base_path / "raw/2017/WADI_attackdata.csv")
	   # Save the cache for next time
	   test17_df.to_parquet(cache_test_file)

	# Remove the long prefix Column Name 
	prefix_to_remove = r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA"

	# Replace the string in all column names
	train17_df.columns = train17_df.columns.str.replace(prefix_to_remove, "", regex=False)
	test17_df.columns = test17_df.columns.str.replace(prefix_to_remove, "", regex=False)


	print(f"Data shape of WADI_14days")
	print(train17_df.shape)


	print(f"Data Info of WADI_14days")
	print(train17_df.info(verbose=True,show_counts=True))
	print(train17_df)


	print(f"Data Shape of WADI_attackdata")
	print(test17_df.shape)


	print(f"Data Info of WADI_attackdata")
	print(test17_df.info(verbose=True, show_counts=True))
	print(test17_df)


	# ### Data Cleaning

	print(f"Perform Data Cleaning")
	# Remove duplicates (if any)
	train17_df.drop_duplicates(inplace=True)
	test17_df.drop_duplicates(inplace=True)


	# Find columns where the count of non-null values is equal to 0
	cols_to_drop = train17_df.columns[train17_df.notnull().sum() == 0]

	# Drop those columns from the DataFrame
	train17_df = train17_df.drop(columns=cols_to_drop)


	# Find columns where the count of non-null values is equal to 0
	cols_to_drop = test17_df.columns[test17_df.notnull().sum() == 0]

	# Drop those columns from the DataFrame
	test17_df = test17_df.drop(columns=cols_to_drop)


	# Check for null values
	pd.set_option('display.max_rows', None)
	train17_df.isnull().sum()
	pd.reset_option('display.max_rows')


	# Check for null values
	pd.set_option('display.max_rows', None)
	test17_df.isnull().sum()
	pd.reset_option('display.max_rows')


	# Remove null values 
	train17_df = train17_df.ffill().bfill()
	test17_df = test17_df.ffill().bfill()


	# Check for null values after remoe
	train17_df.isnull().sum().sum()
	test17_df.isnull().sum().sum()


	# ## Classify data columns into Meta Group, Sensor Group or Actuator Group

	print(f"Classify the data columns into MetaGroup, Sensor Group and Actuator Group - 14days")
	# ==========================================
	# EXPLICIT COLUMN GROUPING
	# ==========================================
	all_cols = train17_df.columns.tolist()

	metadata_cols = []
	actuator_ctrl_cols = []
	sensor_cols = []

	for col in all_cols:
		col_upper = col.upper()

		# Identify Metadata columns
		if any(k in col_upper for k in ["ROW", "DATE", "TIME"]):
			metadata_cols.append(col)
			
		# Identify Continuous Sensor Process Values (_PV and direct flow/pressure metrics)
		elif "_PV" in col_upper or any(
			k in col_upper for k in ["LEAK_DIFF_PRESSURE", "TOTAL_CONS_REQUIRED_FLOW"]
		):
			sensor_cols.append(col)
		
		# Identify Actuators, Alarms, Controller Outputs, and Setpoints (non-continuous sensors)
		elif any(
			k in col_upper
			for k in [
				"STATUS",
				"_AL",
				"_AH",
				"_CO",
				"_SP",
				"_SPEED",
				"PLANT_START_STOP_LOG",
			]
		):
			actuator_ctrl_cols.append(col)
		
		else:
			# Fallback safety: treat anything else as actuator/control if not explicitly a sensor
			actuator_ctrl_cols.append(col)

	print(f"\nGroup Breakdown:")
	print(f" - Metadata Columns: {len(metadata_cols)}")
	print(f" - Actuator Columns: {len(actuator_ctrl_cols)}")
	print(f" - Sensor Columns (Target for K-S): {len(sensor_cols)}")


	print(f"Classify the data columns into MetaGroup, Sensor Group and Actuator Group - attackdata")
	# ==========================================
	# EXPLICIT COLUMN GROUPING
	# ==========================================
	tall_cols = test17_df.columns.tolist()

	tmetadata_cols = []
	tactuator_ctrl_cols = []
	tsensor_cols = []

	for col in tall_cols:
		col_upper = col.upper()

		# Identify Metadata columns
		if any(k in col_upper for k in ["ROW", "DATE", "TIME"]):
			tmetadata_cols.append(col)

		 # Identify Continuous Sensor Process Values (_PV and direct flow/pressure metrics)
		elif "_PV" in col_upper or any(
			k in col_upper for k in ["LEAK_DIFF_PRESSURE", "TOTAL_CONS_REQUIRED_FLOW"]
		):
			tsensor_cols.append(col)
			
		# Identify Actuators, Alarms, Controller Outputs, and Setpoints (non-continuous sensors)
		elif any(
			k in col_upper
			for k in [
				"STATUS",
				"_AL",
				"_AH",
				"_CO",
				"_SP",
				"_SPEED",
				"PLANT_START_STOP_LOG",
			]
		):
			tactuator_ctrl_cols.append(col)

	   
		else:
			# Fallback safety: treat anything else as actuator/control if not explicitly a sensor
			tactuator_ctrl_cols.append(col)

	print(f"\nGroup Breakdown:")
	print(f" - Metadata Columns: {len(tmetadata_cols)}")
	print(f" - Actuator Columns: {len(tactuator_ctrl_cols)}")
	print(f" - Sensor Columns (Target for K-S): {len(tsensor_cols)}")


	# ## Normalize the datasets (Sensor Columns Only)

	print(f"Normalize the Dataset (sensor columns only)-before normalization- 14days")
	print(train17_df.head())


	print(f"Normalize the Dataset (sensor columns only)-after normalization- 14days")
	# data preprocessing
	from sklearn.preprocessing import StandardScaler

	# Standardizing the data (data normalization)
	scaler = StandardScaler()
	scaler = scaler.fit(train17_df[sensor_cols])
	train17_df[sensor_cols] = scaler.transform(train17_df[sensor_cols])
	print(train17_df.head())


	print(f"Normalize the Dataset (sensor columns only)-before normalization- attackdata")
	print(test17_df.head())


	print(f"Normalize the Dataset (sensor columns only)-after normalization- attackdata")
	# data preprocessing
	# from sklearn.preprocessing import StandardScaler

	# Standardizing the data (data normalization)
	scaler = StandardScaler()
	scaler = scaler.fit(test17_df[tsensor_cols])
	test17_df[tsensor_cols] = scaler.transform(test17_df[tsensor_cols])
	print(test17_df.head())


	# ## Add new Column Attack Label and Split Attack data from attackdata dataset

	print(f"Add Attack Label and classify attack label based on specified time ranges - attackdata")
	# Clean column names in case of whitespace artifacts
	test17_df.columns = test17_df.columns.str.strip()

	# ==========================================
	# PARSE TIMESTAMP USING 'Date' AND 'Time' COLUMNS
	# ==========================================
	# Create a temporary datetime series directly from 'DATE' and 'TIME' 
	# without adding a permanent 'DateTime' column to test17_df
	temp_datetime = pd.to_datetime(
		test17_df["Date"].astype(str).str.strip() 
		+ " " 
		+ test17_df["Time"].astype(str).str.strip(),
		format='mixed',
		errors="coerce"
	)

	# ==========================================
	# DEFINE ATTACK TIME RANGES
	# ==========================================
	attack_ranges = [
		("09/10/2017 19:25:00", "09/10/2017 19:50:16"),
		("10/10/2017 10:24:10", "10/10/2017 10:34:00"),
		("10/10/2017 10:55:00", "10/10/2017 11:24:00"),
		("10/10/2017 11:07:46", "10/10/2017 11:12:15"),
		("10/10/2017 11:30:40", "10/10/2017 11:44:50"),
		("10/10/2017 13:39:30", "10/10/2017 13:50:40"),
		("10/10/2017 14:48:17", "10/10/2017 14:59:55"),
		("10/10/2017 14:53:44", "10/10/2017 15:00:32"),
		("10/10/2017 17:40:00", "10/10/2017 17:49:40"),
		("11/10/2017 10:55:00", "11/10/2017 10:56:27"),
		("11/10/2017 11:17:54", "11/10/2017 11:31:20"),
		("11/10/2017 11:36:31", "11/10/2017 11:47:00"),
		("11/10/2017 11:59:00", "11/10/2017 12:05:00"),
		("11/10/2017 12:07:30", "11/10/2017 12:10:52"),
		("11/10/2017 12:16:00", "11/10/2017 12:25:36"),
		("11/10/2017 15:26:30", "11/10/2017 15:37:00"),
	]

	# ==========================================
	# INITIALIZE AND POPULATE ATTACK LABELS
	# ==========================================
	# Initialize attack label column to 1 as default
	test17_df["ATTACK_LABEL"] = 1

	# Iterate through each attack range and mark matching rows as -1
	for start_str, end_str in attack_ranges:
		start_dt = pd.to_datetime(start_str, format="%d/%m/%Y %H:%M:%S")
		end_dt = pd.to_datetime(end_str, format="%d/%m/%Y %H:%M:%S")

		mask = (temp_datetime >= start_dt) & (temp_datetime <= end_dt)
		test17_df.loc[mask, "ATTACK_LABEL"] = -1

	# ==========================================
	# VERIFICATION
	# ==========================================
	print("\nLabel distribution summary:")
	print(test17_df["ATTACK_LABEL"].value_counts())
	print(f"\nFinal DataFrame shape: {test17_df.shape}")
	print(f"Last column name: {test17_df.columns[-1]}")


	print(f"Data info of attackdata after adding Attack Label")
	test17_df.info(verbose=True, show_counts=True)
	print(test17_df)


	print(f"Separate the attackdata into 2 datasets: 1 with normal data and 1 with attack data")
	# ==========================================
	# SEPARATE DATASET BY ATTACK LABEL
	# ==========================================

	# Filter rows where ATTACK_LABEL is 1 (Normal data)
	test17_normal_df = test17_df[test17_df["ATTACK_LABEL"] == 1].copy()

	# Filter rows where ATTACK_LABEL is -1 (Attack data)
	test17_attack_df = test17_df[test17_df["ATTACK_LABEL"] == -1].copy()

	# ==========================================
	# VERIFICATION
	# ==========================================
	print(f"Original DataFrame shape: {test17_df.shape}")
	print(f"Normal DataFrame shape (`test17_normal_df`): {test17_normal_df.shape}")
	print(f"Attack DataFrame shape (`test17_attack_df`): {test17_attack_df.shape}")


	# ## Task 1a Compute K-S Statistic 

	print(f"Task 1a Compute K-S Statistic excluding data under attack (Sensor Columns)")
	# ==========================================
	# CUSTOM K-S STATISTIC FUNCTION
	# ==========================================
	def compute_ks_statistic(sample1, sample2):
		"""Computes the two-sample Kolmogorov-Smirnov statistic
		"""
		s1 = np.sort(sample1)
		s2 = np.sort(sample2)

		data_all = np.unique(np.concatenate([s1, s2]))

		# Compute Empirical Cumulative Distribution Functions (ECDFs)
		ecdf1 = np.searchsorted(s1, data_all, side="right") / len(s1)
		ecdf2 = np.searchsorted(s2, data_all, side="right") / len(s2)

		# Find the maximum absolute vertical distance between ECDFs
		ks_stat = np.max(np.abs(ecdf1 - ecdf2))
		return ks_stat


	# ==========================================
	# ITERATE AND COMPARE SENSOR GROUP COLUMN-BY-COLUMN
	# ==========================================
	ks_results = {}

	for col in sensor_cols:
		train_sample = train17_df[col].values
		test_normal_sample = test17_normal_df[col].values

		# Compute K-S statistic between train17 baseline and normal test17 subset
		ks_val = compute_ks_statistic(train_sample, test_normal_sample)
		ks_results[col] = ks_val

	# Summarize results into a DataFrame
	ks_df = pd.DataFrame(
		list(ks_results.items()), columns=["Sensor_Column", "KS_Statistic"]
	)
	print("\nSensor List from Highest Distribution Shift (K-S Statistic):")
	print(ks_df.sort_values(by="KS_Statistic", ascending=False).head(100))


	# ## Task 1b Compute System State and Percentages

	print(f"Task 1b Compute the set of System States and their percentages")
	# ==========================================
	# EXTRACT SYSTEM STATES AS SETS
	# ==========================================
	# Clean missing values to ensure consistent tuple conversion
	train_actuators = train17_df[actuator_ctrl_cols].fillna(0)
	test_normal_actuators = test17_normal_df[actuator_ctrl_cols].fillna(0)

	# Convert each row into a hashable tuple representing the system state at that time step
	train_tuples = [tuple(x) for x in train_actuators.to_numpy()]
	test_normal_tuples = [tuple(x) for x in test_normal_actuators.to_numpy()]

	set_train = set(train_tuples)
	set_test_normal = set(test_normal_tuples)

	# Count unique system states
	count_train = len(set_train)
	count_test_normal = len(set_test_normal)

	print(f"\nUnique system states in train17_df - WADI_14days: {count_train}")
	print(f"Unique system states in test17_normal_df - WADI_attackdata: {count_test_normal}")

	print(f"\nCompute the Frequencies and percentages of System State, list 20 top percentages")

	# ==========================================
	# CALCULATE FREQUENCIES AND PERCENTAGES
	# ==========================================
	# Count occurrences of each unique system state configuration
	train_counts = Counter(train_tuples)
	test_normal_counts = Counter(test_normal_tuples)

	total_train_rows = len(train_tuples)
	total_test_rows = len(test_normal_tuples)

	# Retrieve top 20 most frequent states (state tuple, absolute frequency count)
	top_train_states = train_counts.most_common(20)
	top_test_normal_states = test_normal_counts.most_common(20)

	# ==========================================
	# DISPLAY TOP 20 RESULTS
	# ==========================================
	print("\n" + "=" * 80)
	print("TOP 20 MOST FREQUENT SYSTEM STATES AND PERCENTAGES IN TRAIN17_DF (WADI_14days)")
	print("=" * 80)
	for rank, (state, freq) in enumerate(top_train_states, start=1):
		percentage = (freq / total_train_rows) * 100
		print(f"Rank {rank:2d}: Count = {freq:6d} | Percentage = {percentage:5.10f}%")
		# Display the column-value pairs for this system state
		state_dict = dict(zip(actuator_ctrl_cols, state))
		for col, val in state_dict.items():
			print(f"    {col}: {val}")
		print("-" * 80)


	print("\n" + "=" * 80)
	print("TOP 20 MOST FREQUENT SYSTEM STATES AND PERCENTAGES IN TEST17_NORMAL_DF - WADI_attackdata exclude attack data")
	print("=" * 80)
	for rank, (state, freq) in enumerate(top_test_normal_states, start=1):
		percentage = (freq / total_test_rows) * 100
		print(f"Rank {rank:2d}: Count = {freq:6d} | Percentage = {percentage:5.10f}%")
		# Display the column-value pairs for this system state
		state_dict = dict(zip(actuator_ctrl_cols, state))
		for col, val in state_dict.items():
			print(f"    {col}: {val}")
		print("-" * 80)

