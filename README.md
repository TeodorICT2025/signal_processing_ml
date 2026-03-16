# Sine Wave Detection with Logistic Regression

This project generates a synthetic dataset for detecting whether a noisy signal contains a sine wave within a specific frequency band. The signals are analyzed using simple signal-processing features, and the resulting dataset can be used to train a machine learning classifier such as Logistic Regression.

The goal is to determine whether a signal contains a sinusoidal component between **40 Hz and 60 Hz**.

---

## Signal Generation

Each signal is composed of **256 samples** recorded at a **sampling frequency of 200 Hz**.

Signals belong to two classes:

### Class 0 (Negative)

Signals that **do not contain a sine wave in the target band**.

They consist of:

- Gaussian noise with random standard deviation
- Sometimes a sine wave outside the target band ( *hard negatives*)

Hard negatives make the classification problem harder by introducing signals that still contain periodic components but **not in the desired frequency interval**.

Two frequency ranges are used for hard negatives:

- **5 – 35 Hz**
- **65 – 120 Hz**

---

### Class 1 (Positive)

Signals that **contain a sine wave between 40 Hz and 60 Hz**.

Each signal is generated as:

signal = A sin(2πft + φ) + noise

where:

- `A` = random amplitude
- `f` = random frequency in the desired band
- `φ` = random phase
- Gaussian noise is added to simulate measurement noise

---

## Feature Extraction

For each signal, three features are extracted.

### 1. Signal Energy

Total signal energy:

\[
x_1 = \sum x^2
\]

This measures overall signal strength.

---

### 2. Signal Variance

Variance of the signal:

\[
x_2 = Var(x)
\]

This captures signal spread and noise characteristics.

---

### 3. Band Power (FFT Feature)

The signal is transformed into the frequency domain using the **Fast Fourier Transform (FFT)**.

The power spectrum is computed and the energy inside the **40–60 Hz band** is summed:

\[
x_3 = \sum |X(f)|^2
\]

where:

- `X(f)` is the FFT of the signal
- only frequencies in the target band are included

This feature captures how much energy exists in the desired frequency interval.

---

## Dataset Structure

The generated dataset contains:

- **2000 samples total**
- **1000 negative samples**
- **1000 positive samples**

Each row has the format:

```
x1, x2, x3, y
```

where:

- `x1` = signal energy
- `x2` = signal variance
- `x3` = FFT band power (40–60 Hz)
- `y` = label  
  - `0` = no sine wave in band  
  - `1` = sine wave present in band

Example row:

```
523.1, 2.31, 95.8, 1
```

---

## Dataset Generation

The dataset is generated using the script:

```
generate_data.py
```

Run:

```
python generate_data.py
```

This produces:

```
data.csv
```

with the following structure:

```
x1,x2,x3,y
...
```

---

## Reproducibility

A fixed random seed is used:

```
rng = np.random.default_rng(42)
```

This ensures the dataset generation is **fully reproducible**.

---

## Purpose of the Dataset

This dataset is designed for experimenting with:

- Logistic Regression
- Binary classification
- Signal processing + machine learning pipelines
- Feature engineering from time-series signals

It demonstrates how **frequency-domain features (FFT)** can help detect hidden signals inside noisy measurements.

---

## Possible Extensions

Some possible improvements include:

- adding more frequency bands
- using additional features (spectral centroid, peak frequency)
- applying band-pass filters before FFT
- testing other classifiers such as SVM or neural networks

---

## Technologies Used

- Python
- NumPy
- Fast Fourier Transform (FFT)
