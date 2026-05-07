

## Progress So Far

### 1. Dataset Understanding

* Using NASA CMAPSS dataset (FD001)

* Dataset contains multiple engine trajectories

* Each engine has:

  * Engine ID
  * Cycle number
  * 3 operational settings
  * 21 sensor readings

* Each row represents one cycle of engine operation

---

### 2. Data Loading

* Loaded `train_FD001.txt` using Pandas

* Handled extra spaces in dataset by removing empty columns

* Assigned proper column names:

  * engine_id, cycle
  * operational settings
  * sensor_1 to sensor_21

* Verified dataset shape:

  * (20631, 26)

---

### 3. RUL (Remaining Useful Life) Creation

* Computed maximum cycle for each engine

* Calculated RUL using:

  RUL = max_cycle - current_cycle

* Verified correctness:

  * RUL decreases over time
  * Final cycle of each engine has RUL = 0

---

### 4. RUL Clipping

* Applied upper cap of 125 on RUL values

* Reason:

  * Prevent large early-cycle values from dominating training
  * Focus model on meaningful degradation region

* Verified:

  * Max RUL = 125
  * Distribution looks stable

---

## Current Status

* Data loading pipeline is complete
* RUL target variable successfully created and validated

---

## Next Steps

* Feature selection (remove irrelevant sensors)
* Data normalization (scaling)
* Sequence generation for LSTM model
* Model building and training
* Evaluation and visualization
