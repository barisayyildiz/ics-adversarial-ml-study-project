#!/usr/bin/env python
# coding: utf-8

# # Study Project: Adversarial Machine Learning
# ## Part I: Statistical Analysis and Generation of Process Data
# ## Group 15
# ## SIN Wang Yat
# 
# ### Task 3: Generate of Synthetic Dataset 
# 
# #### Inputs & Value Descriptions
# - Input 1 (Dataset Selection): Choose WADI_14days.csv or WADI_attackdata.csv
# - Input 2 (Row Sub-sampling Percentage in 1-100%): Randomly select a fraction of sample rows from the chosen dataset
# - Input 3 (Normalization Type Selection): Choose StandardScaler or MinMaxScaler 
# - Input 4 (Synthetic Generation Percentage in 1-1000%): Indicate the amount of synthetic samples to generate
# - Input 5 (Number of Nearest Neighbors k in 2-5): Specify the number of neighbors used for Manhattan distanc
# 
# #### Output Data & CSV Dataset 
# - Output Data is automatically exported and saved as synthetic_data.csv under the target directory Path.home() /Projects/P1/data/processed/

# ### Libray Import and Data Loading

print(f"Start: Import Library and Load the datasets")
# Library Import
import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timedelta

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


	print(f"Data Shape of WADI_14days")
	print(train17_df.shape)


	print(f"Data info of WADI_14days")
	print(train17_df.info(verbose=True,show_counts=True))
	print(train17_df)


	print(f"Data Share of WADI_attackdata")
	print(test17_df.shape)


	print(f"Data Info of WADI_attackdata")
	print(test17_df.info(verbose=True, show_counts=True))
	print(test17_df)


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
	print(f" - Sensor Columns: {len(sensor_cols)}")


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
	print(f" - Sensor Columns: {len(tsensor_cols)}")


	# ## Add Attack Lablel to  Attack data

	print(f"Add Attack Label and Classify the attack label in WADI_attackdata")
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
	# 3. DEFINE ATTACK TIME RANGES
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
	# 4. INITIALIZE AND POPULATE ATTACK LABELS
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
	# 5. VERIFICATION
	# ==========================================
	print("\nLabel distribution summary:")
	print(test17_df["ATTACK_LABEL"].value_counts())
	print(f"\nFinal DataFrame shape: {test17_df.shape}")
	print(f"Last column name: {test17_df.columns[-1]}")


	# ## Task 3: Generate the Synthetic Data from selected dataset

	def run_synthetic_generation_cli():
		print("=" * 60)
		print("TASK 3: SYNTHETIC DATASET GENERATION")
		print("=" * 60)
		
		 
		# ==========================================
		# INPUT 1: DATASET SELECTION
		# ==========================================
		while True:
			ds_input = input("Select dataset [1: train17_df, 2: test17_df] (Enter 1 or 2): ").strip()
			if ds_input == "1":
				raw_df = train17_df.copy()
				file_name = "WADI_14days.csv"
				break
			elif ds_input == "2":
				raw_df = test17_df.copy()
				file_name = "WADI_attackdata.csv"
				break
			else:
				print("Invalid selection. Please enter '1' or '2'.")
				
		print(f"-> Selected Dataset: ({file_name})")

		# ==========================================
		# INPUT 2: PERCENTAGE OF ROWS TO RANDOMLY SAMPLE (1 - 100%) - Control the processing time
		# ==========================================
		while True:
			try:
				row_percentage = int(input("Enter percentage of sample rows to select from dataset (Integer range: 1 - 100%): ").strip())
				if 1 <= row_percentage <= 100:
					break
				else:
					print("Error: Percentage must be between 1 and 100.")
			except ValueError:
				print("Error: Invalid input. Please enter a valid integer.")
				
		print(f"-> Selected Row Sub-sample Percentage: {row_percentage}%")
		
		# Randomly select sample rows as selected_sample_df
		n_sample_rows = int(len(raw_df) * (row_percentage / 100.0))
		selected_sample_df = raw_df.sample(n=n_sample_rows, random_state=42).reset_index(drop=True)
		print(f"-> 'selected_sample_df' created with {len(selected_sample_df)} rows out of {len(raw_df)}.")

		# ==========================================
		# INPUT 3: NORMALIZATION SCALER SELECTION
		# ==========================================
		while True:
			scaler_input = input("Select scaler type [1: StandardScaler, 2: MinMaxScaler] (Enter 1 or 2): ").strip()
			if scaler_input == "1":
				scaler_type = "standard"
				break
			elif scaler_input == "2":
				scaler_type = "minmax"
				break
			else:
				print("Invalid selection. Please enter '1' or '2'.")
				
		print(f"-> Selected Scaler: {scaler_type.upper()}")

		# ==========================================
		# INPUT 4: SYNTHETIC GENERATION PERCENTAGE (1 - 1000%)
		# ==========================================
		while True:
			try:
				percentage = int(input("Enter synthetic sample generation percentage (Integer range: 1 - 1000%): ").strip())
				if 1 <= percentage <= 1000:
					break
				else:
					print("Error: Percentage must be between 1 and 1000.")
			except ValueError:
				print("Error: Invalid input. Please enter a valid integer.")
				
		print(f"-> Selected Synthetic Percentage: {percentage}%")

		# ==========================================
		# INPUT 5: K NEIGHBORS (2 - 5)
		# ==========================================
		while True:
			try:
				k = int(input("Enter number of nearest neighbors k (Integer range: 2 - 5): ").strip())
				if 2 <= k <= 5:
					break
				else:
					print("Error: k must be between 2 and 5.")
			except ValueError:
				print("Error: Invalid input. Please enter a valid integer.")
				
		print(f"-> Selected k: {k}")

		sensor_data = selected_sample_df[sensor_cols].ffill().bfill().fillna(0).to_numpy(dtype=float)

		# Start timestamp and time
		generation_start_time = datetime.now()
		print(f"-> Generation Start Timestamp: {generation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
		
		# ==========================================
		# CUSTOM NORMALIZATION FROM SCRATCH
		# ==========================================
		if scaler_type == "standard":
			means = np.mean(sensor_data, axis=0)
			stds = np.std(sensor_data, axis=0)
			stds[stds == 0] = 1.0
			normalized_sensors = (sensor_data - means) / stds
		else:
			mins = np.min(sensor_data, axis=0)
			maxs = np.max(sensor_data, axis=0)
			ranges = maxs - mins
			ranges[ranges == 0] = 1.0
			normalized_sensors = (sensor_data - mins) / ranges

		working_df = selected_sample_df.copy()
		working_df[sensor_cols] = normalized_sensors

		# ==========================================
		# Compute and Select K-NN
		# ==========================================
		print("Computing k-nearest neighbors")
		n_samples = len(normalized_sensors)
		k_neighbors = np.empty((n_samples, k), dtype=int)
		
		for i in range(n_samples):
			dists = np.sum(np.abs(normalized_sensors - normalized_sensors[i]), axis=1)
			dists[i] = np.inf
			k_neighbors[i] = np.argsort(dists)[:k]
			
			if (i + 1) % 20000 == 0 or (i + 1) == n_samples:
				print(f"Processed {i + 1} / {n_samples} rows...")

		print("Finished k-NN computation successfully.")

		# ==========================================
		# SEQUENTIAL SYNTHETIC SAMPLE GENERATION
		# ==========================================
		n_real_samples = len(working_df)
		n_synthetic_samples = int(n_real_samples * (percentage / 100.0))
		chunk_size = 10000
		
		print(f"Generating {n_synthetic_samples} synthetic samples sequentially with chunk_size={chunk_size}...")

		synthetic_rows = []
		generated_count = 0

		while generated_count < n_synthetic_samples:
			current_chunk_size = min(chunk_size, n_synthetic_samples - generated_count)
			
			for _ in range(current_chunk_size):
				idx = random.randint(0, n_real_samples - 1)
				base_sample = working_df.iloc[idx]
				
				neighbor_idx = random.choice(k_neighbors[idx])
				neighbor_sample = working_df.iloc[neighbor_idx]
				
				alpha = random.random()
				synthetic_row = base_sample.copy()
				synthetic_row[sensor_cols] = base_sample[sensor_cols] + alpha * (neighbor_sample[sensor_cols] - base_sample[sensor_cols])
				synthetic_rows.append(synthetic_row)
				
			generated_count += current_chunk_size
			print(f"Generated {generated_count} / {n_synthetic_samples} synthetic samples...")

		synthetic_data_df = pd.DataFrame(synthetic_rows)

		# ==========================================
		# POPULATE ROW, DATE, AND TIME COLUMNS
		# ==========================================
		# 1. Row column starts from integer 1 and sequentially increases by 1
		synthetic_data_df["Row"] = range(1, len(synthetic_data_df) + 1)

		# RESET DF INDEX
		synthetic_data_df = synthetic_data_df.reset_index(drop=True)

		# 2 & 3. Date starts at "10/10/2017" (DD/MM/YYYY) and increases daily; 
		# Time starts at "00:00:00" (HH:MM:SS) and increases by 1 second per row.
		base_datetime = datetime(2017, 10, 10, 0, 0, 0)
		datetimes = [base_datetime + timedelta(seconds=i) for i in range(len(synthetic_data_df))]
		
		synthetic_data_df["Date"] = [dt.strftime("%d/%m/%Y") for dt in datetimes]
		synthetic_data_df["Time"] = [dt.strftime("%H:%M:%S") for dt in datetimes]
		
		# Reorder columns to place Row, Date, Time at the front if they exist or were created
		cols_order = ["Row", "Date", "Time"] + [c for c in synthetic_data_df.columns if c not in ["Row", "Date", "Time"]]
		synthetic_data_df = synthetic_data_df[cols_order]


		# ==========================================
		# SAVE TO PROCESSED DIRECTORY
		# ==========================================
		output_dir = base_path / "processed"
		output_dir.mkdir(parents=True, exist_ok=True)
		
		output_file = output_dir / "synthetic_data.csv"
		synthetic_data_df.to_csv(output_file, index=False)
		
		# Track overall end timestamp and calculate duration
		generation_end_time = datetime.now()
		duration_minutes = (generation_end_time - generation_start_time).total_seconds() / 60.0
		
		print("\n" + "=" * 60)
		print("SUCCESSFULLY GENERATED AND SAVED SYNTHETIC DATASET")
		print("=" * 60)
		print(f"Start Timestamp : {generation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
		print(f"End Timestamp   : {generation_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
		print(f"Duration        : {duration_minutes:.4f} minutes")
		print(f"Output File Path: {output_file}")
		print(f"Synthetic Dataset Shape: {synthetic_data_df.shape}")
		
		return synthetic_data_df


	# To run the interactive CLI program, uncomment the line below:
	synthetic_data_df = run_synthetic_generation_cli()


	print('Data info of synthetic_data_df - generated')
	print(synthetic_data_df.info(verbose=True,show_counts=True))
	print(synthetic_data_df)




