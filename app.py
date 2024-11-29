# import numpy as np
# from flask import Flask, request, render_template
# import pickle

# flask_app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods=["POST"])
# def predict():
   
#     hex_feature = request.form['hexa'] 
#     features = [hex_feature] 


#     prediction = model.predict([features])  

#     return render_template("index.html", prediction_text="The algorithm is {}".format(prediction[0]))

# if __name__ == "__main__":
#     flask_app.run(debug=True)

from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
import pickle  # Assuming you're using joblib to load your model
from collections import Counter
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from model import scaler_function
from scipy.stats import skew, kurtosis, chisquare, kstest, entropy as scipy_entropy
from numpy.fft import fft
# from model import label_encoder1_function
from sklearn.preprocessing import LabelEncoder
import itertools
from sklearn.preprocessing import StandardScaler

flask_app = Flask(__name__)

# Load your machine learning model (replace 'your_model.pkl' with the correct path)
model = pickle.load(open("model.pkl", "rb"))

# Convert hex string to bytes
def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

# Calculate entropy of byte data
def calculate_entropy(data):
    if (len(data) == 0):
        return 0
    entropy = 0
    data_len = len(data)
    counter = Counter(data)
    for count in counter.values():
        probability = count / data_len
        entropy -= probability * math.log2(probability)
    return entropy

# Top-N frequencies
def top_n_frequencies(data, n=5):
    freq_dist = Counter(data)
    most_common = freq_dist.most_common(n)
    return {f'top_{i+1}_freq': count for i, (char, count) in enumerate(most_common)}

# Byte-level statistics
def byte_statistics(data):
    if len(data) == 0:  # Check if the input data is empty
        return {
            'mean_byte_value': 0,
            'median_byte_value': 0,
            'variance_byte_value': 0,
            'std_dev_byte_value': 0,
            'skewness_byte_value': 0,
            'kurtosis_byte_value': 0,
        }
    
    byte_values = np.array(list(data))
    stats = {
        'mean_byte_value': np.mean(byte_values),
        'median_byte_value': np.median(byte_values),
        'variance_byte_value': np.var(byte_values),
        'std_dev_byte_value': np.std(byte_values),
        'skewness_byte_value': skew(byte_values),
        'kurtosis_byte_value': kurtosis(byte_values),
    }
    return stats

# Frequency statistics
def frequency_statistics(data):
    freq_dist = Counter(data)
    freqs = np.array(list(freq_dist.values()))
    stats = {
        'max_byte_freq': np.max(freqs),
        'min_byte_freq': np.min(freqs),
        'range_byte_freq': np.max(freqs) - np.min(freqs),
        'std_dev_byte_freq': np.std(freqs),
        'entropy_byte_freq': scipy_entropy(list(freqs))  # Convert dict_values to list
    }
    return stats

# N-gram (Bigram, Trigram, Quadgram) statistics
def ngram_statistics(data, n=2):
    if len(data) < n:  # Guard condition to check if there's enough data
        return {f'{n}gram_max_freq': 0, f'{n}gram_min_freq': 0, f'{n}gram_range_freq': 0, f'{n}gram_std_dev_freq': 0, f'{n}gram_entropy_freq': 0}
    
    ngrams = Counter([tuple(data[i:i+n]) for i in range(len(data)-n+1)])
    freqs = np.array(list(ngrams.values()))
    if freqs.size == 0:  # Additional check for empty frequencies
        return {f'{n}gram_max_freq': 0, f'{n}gram_min_freq': 0, f'{n}gram_range_freq': 0, f'{n}gram_std_dev_freq': 0, f'{n}gram_entropy_freq': 0}
    
    stats = {
        f'{n}gram_max_freq': np.max(freqs),
        f'{n}gram_min_freq': np.min(freqs),
        f'{n}gram_range_freq': np.max(freqs) - np.min(freqs),
        f'{n}gram_std_dev_freq': np.std(freqs),
        f'{n}gram_entropy_freq': scipy_entropy(list(freqs))
    }
    return stats

# Calculate autocorrelation at a given lag
def calculate_autocorrelation(data, lag):
    byte_values = np.array(list(data))  # Convert byte data into a list of integers
    n = len(byte_values)
    mean = np.mean(byte_values)
    autocorr = np.correlate(byte_values - mean, byte_values - mean, mode='full')[n - 1:] / np.var(byte_values) / n
    return autocorr[lag] if lag < len(autocorr) else 0

# FFT statistics
def fft_statistics(data):
    byte_values = np.array(list(data), dtype=np.float64)  # Ensure proper data type
    if byte_values.size == 0:  # Guard condition for empty byte data
        return {
            'fft_mean_magnitude': 0,
            'fft_std_dev_magnitude': 0,
            'fft_max_magnitude': 0,
            'fft_min_magnitude': 0,
            'fft_median_magnitude': 0,
        }
    
    fft_vals = np.abs(fft(byte_values))
    return {
        'fft_mean_magnitude': np.mean(fft_vals),
        'fft_std_dev_magnitude': np.std(fft_vals),
        'fft_max_magnitude': np.max(fft_vals),
        'fft_min_magnitude': np.min(fft_vals),
        'fft_median_magnitude': np.median(fft_vals),
    }

# Calculate compression ratio (using gzip for example)
def compression_ratio(data):
    import gzip
    compressed = gzip.compress(data)
    return len(compressed) / len(data)

# Hamming weight of bytes
def average_hamming_weight(data):
    hamming_weight = sum(bin(byte).count('1') for byte in data)
    return hamming_weight / len(data)

# Run tests for randomness (based on consecutive bytes)
def runs_test(data):
    runs = 1
    for i in range(1, len(data)):
        if data[i] != data[i-1]:
            runs += 1
    return runs

# Chi-square test statistic
def chi_square_test(data):
    freq_dist = Counter(data)
    observed = np.array(list(freq_dist.values()))
    expected = np.full(len(observed), np.mean(observed))
    chi2, _ = chisquare(observed, expected)
    return chi2

# Kolmogorov-Smirnov test statistic (against uniform distribution)
def ks_test(data):
    byte_values = np.array(list(data))
    d_stat, _ = kstest(byte_values, 'uniform', args=(np.min(byte_values), np.max(byte_values)))
    return d_stat

# Serial correlation
def serial_correlation(data):
    byte_values = np.array(list(data))
    return np.corrcoef(byte_values[:-1], byte_values[1:])[0, 1]

# Percentage of printable ASCII characters
def printable_ascii_percentage(data):
    printable = sum(32 <= byte <= 126 for byte in data)
    return printable / len(data)

# Extract features from the ciphertext
def extract_features(ciphertext_hex, features):
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    features['length'] = len(ciphertext_bytes)

    # Byte-level statistics
    byte_stats = byte_statistics(ciphertext_bytes)
    features.update(byte_stats)

    # Entropy
    features['entropy'] = calculate_entropy(ciphertext_bytes)

    # Frequency distribution statistics
    freq_stats = frequency_statistics(ciphertext_bytes)
    features.update(freq_stats)

    # Bigram, Trigram, and Quadgram statistics
    for n in [2, 3, 4]:
        ngram_stats = ngram_statistics(ciphertext_bytes, n=n)
        features.update(ngram_stats)

    # Autocorrelation
    for lag in [1, 2, 5, 10]:
        features[f'autocorr_lag_{lag}'] = calculate_autocorrelation(ciphertext_bytes, lag)

    # FFT statistics
    fft_stats = fft_statistics(ciphertext_bytes)
    features.update(fft_stats)

    # Compression ratio
    features['compression_ratio'] = compression_ratio(ciphertext_bytes)

    # Hamming weight
    features['avg_hamming_weight'] = average_hamming_weight(ciphertext_bytes)

    # Runs test statistic
    features['runs_test'] = runs_test(ciphertext_bytes)

    # Chi-square test
    features['chi_square_test'] = chi_square_test(ciphertext_bytes)

    # Kolmogorov-Smirnov test
    features['ks_test_stat'] = ks_test(ciphertext_bytes)

    # Serial correlation
    features['serial_correlation'] = serial_correlation(ciphertext_bytes)

    # Percentage of printable ASCII characters
    features['printable_ascii_percentage'] = printable_ascii_percentage(ciphertext_bytes)

    # Additional byte-level statistics
    byte_values = np.array(list(ciphertext_bytes))
    features['avg_byte_value_change'] = np.mean(np.abs(np.diff(byte_values)))
    features['median_abs_dev_byte_values'] = np.median(np.abs(byte_values - np.median(byte_values)))
    features['iqr_byte_values'] = np.percentile(byte_values, 75) - np.percentile(byte_values, 25)
    features['coef_variation_byte_values'] = np.std(byte_values) / np.mean(byte_values) if np.mean(byte_values) != 0 else 0
    features['pct_bytes_above_mean'] = np.sum(byte_values > np.mean(byte_values)) / len(byte_values)

    # Entropy of byte value gaps
    byte_value_gaps = np.abs(np.diff(byte_values))
    features['entropy_byte_value_gaps'] = scipy_entropy(list(Counter(byte_value_gaps).values()))  # Convert dict_values to list

    return features

# Extract IV and infer mode of operation
def extract_iv_and_infer_mode(ciphertext_hex, features, block_size=16):
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    iv = ciphertext_bytes[:block_size]
    features['iv'] = iv

    if len(ciphertext_bytes) % block_size != 0:
        features['mode'] = 'Unknown or Stream Cipher'
    else:
        blocks = [ciphertext_bytes[i:i + block_size] for i in range(0, len(ciphertext_bytes), block_size)]
        if len(blocks) != len(set(blocks)):
            features['mode'] = 'ECB'
        else:
            features['mode'] = 'CBC or other block mode'

    return features

def byte_value_range(data):
    return np.ptp(data)

def mode_of_byte_values(data):
    return Counter(data).most_common(1)[0][0]

def frequency_of_mode_byte_value(data):
    return Counter(data).most_common(1)[0][1] / len(data)

def byte_value_histogram(data, bins=256):
    hist, _ = np.histogram(data, bins=bins, range=(0, 255))
    return hist.tolist()

def byte_value_percentiles(data):
    return np.percentile(data, [25, 50, 75]).tolist()

def entropy_of_byte_value_differences(data):
    if (len(data) != 0):
        differences = np.diff(data)
        return calculate_entropy(differences)
    else:
        return 0

def frequency_of_byte_value_differences(data):
    differences = np.diff(data)
    return dict(Counter(differences))

def longest_increasing_subsequence(data):
    n = len(data)
    if n == 0:
        return 0
    lengths = [1] * n
    for i in range(1, n):
        for j in range(i):
            if data[i] > data[j] and lengths[i] < lengths[j] + 1:
                lengths[i] = lengths[j] + 1
    return max(lengths)

def longest_decreasing_subsequence(data):
    return longest_increasing_subsequence([-x for x in data])

def run_length_encoding(data):
    return [(len(list(group)), name) for name, group in itertools.groupby(data)]

def byte_value_transition_matrix(data):
    matrix = np.zeros((256, 256), dtype=int)
    for i in range(len(data) - 1):
        matrix[data[i]][data[i+1]] += 1
    return matrix.tolist()

def frequency_of_byte_value_n_grams(data, n):
    n_grams = zip(*[data[i:] for i in range(n)])
    return dict(Counter(n_grams))

def entropy_of_byte_value_n_grams(data, n):
    n_gram_freq = frequency_of_byte_value_n_grams(data, n)
    return scipy_entropy(list(n_gram_freq.values()))

def byte_value_autocorrelation_function(data, nlags=50):
    result = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    result = result[result.size//2:]
    return result[:nlags].tolist()

def byte_value_power_spectrum(data):
    return np.abs(np.fft.fft(data))**2

# Updated extract_features function
def extract_features1_new(ciphertext_hex, features):
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    byte_values = np.array(list(ciphertext_bytes))

    # Existing feature extraction (keep all the existing feature extractions)

    # New feature extractions
    features['byte_value_range'] = byte_value_range(byte_values)
    features['mode_byte_value'] = mode_of_byte_values(byte_values)
    features['freq_mode_byte_value'] = frequency_of_mode_byte_value(byte_values)
    features['byte_value_histogram'] = byte_value_histogram(byte_values)
    features['byte_value_percentiles'] = byte_value_percentiles(byte_values)
    features['entropy_byte_value_diff'] = entropy_of_byte_value_differences(byte_values)
    features['freq_byte_value_diff'] = frequency_of_byte_value_differences(byte_values)
    features['longest_increasing_subseq'] = longest_increasing_subsequence(byte_values)
    features['longest_decreasing_subseq'] = longest_decreasing_subsequence(byte_values)
    features['run_length_encoding'] = run_length_encoding(byte_values)
    features['byte_value_transition_matrix'] = byte_value_transition_matrix(byte_values)

    for n in [2, 3, 4]:
        features[f'freq_byte_value_{n}grams'] = frequency_of_byte_value_n_grams(byte_values, n)
        features[f'entropy_byte_value_{n}grams'] = entropy_of_byte_value_n_grams(byte_values, n)

    features['byte_value_acf'] = byte_value_autocorrelation_function(byte_values)
    features['byte_value_power_spectrum'] = byte_value_power_spectrum(byte_values).tolist()

    return features

scaler = StandardScaler()
def scaler_function():
    return scaler


label_encoder1 = LabelEncoder()
def label_encoder1_function():
    return label_encoder1


@flask_app.route("/")
def Home():
    return render_template("index.html")

# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     # Get the hexadecimal data from the form
#     hex_data = request.form.get('hexa')  # form data instead of json
    
#     if hex_data is None:
#         return jsonify({'error': 'No hex_data provided'}), 400
    
    
#     try:
#         hex_data = ''.join(hex_data.split())
#         print(hex_data)
#         features = {}
#         # Step 1: Extract features from the hexadecimal data
#         features = extract_features(hex_data, features=features)
        
#         # Step 2: Extract IV and infer mode of operation
#         features = extract_iv_and_infer_mode(hex_data, features)
#         df1 = pd.DataFrame([features])
        
#         features = {}
#         features = extract_features1_new(hex_data, features=features)
#         df2 = pd.DataFrame([features])
        
#         features_df = pd.concat([df1, df2], axis=1)

#         features_df.drop(columns=['iv', 'byte_value_histogram', 'byte_value_percentiles', 'freq_byte_value_diff', 'run_length_encoding', 'byte_value_transition_matrix', 'freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams', 'byte_value_acf', 'byte_value_power_spectrum'], inplace=True)
#         X = features_df
#         label_encoder1 = label_encoder1_function()
#         X['mode'] = label_encoder1.transform(X[['mode']].values.flatten())
#         # Step 3: Make predictions using your ML model
#         scaler = scaler_function()
#         exclude_columns = ['mode']
#         columns_to_scale = X.columns.difference(exclude_columns)
#         X[columns_to_scale] = scaler.transform(X[columns_to_scale])

#         print(X)
#         prediction = model.predict(X)
#         print(prediction)
#         return render_template("index.html", prediction_text=f"The algorithm is {prediction}")
    
#     # except ValueError:
#     #     return jsonify({'error': 'Invalid hexadecimal data'}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@flask_app.route('/predict', methods=['POST'])
def predict():
    hex_data = request.form.get('hexa')
    
    if hex_data is None:
        return jsonify({'error': 'No hex_data provided'}), 400
    
    try:
        hex_data = ''.join(hex_data.split())
        print("Hex data:", hex_data)
        
        features = {}
        features = extract_features(hex_data, features=features)
        print("Features extracted (1):", features)
        
        features = extract_iv_and_infer_mode(hex_data, features)
        print("IV and mode inferred:", features)
        
        df1 = pd.DataFrame([features])
        print("df1 shape:", df1.shape)
        
        features = {}
        features = extract_features1_new(hex_data, features=features)
        print("Features extracted (2):", features)
        
        df2 = pd.DataFrame([features])
        print("df2 shape:", df2.shape)
        
        features_df = pd.concat([df1, df2], axis=1)
        print("Combined features shape:", features_df.shape)

        features_df.drop(columns=['iv', 'byte_value_histogram', 'byte_value_percentiles', 'freq_byte_value_diff', 'run_length_encoding', 'byte_value_transition_matrix', 'freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams', 'byte_value_acf', 'byte_value_power_spectrum'], inplace=True)
        X = features_df
        print("X shape after drop:", X.shape)

        label_encoder1 = label_encoder1_function()
        X['mode'] = label_encoder1.transform(X[['mode']].values.flatten())
        print("X['mode'] after transform:", X['mode'])

        scaler = scaler_function()
        exclude_columns = ['mode']
        columns_to_scale = X.columns.difference(exclude_columns)
        X[columns_to_scale] = scaler.transform(X[columns_to_scale])
        print("X shape after scaling:", X.shape)

        print("X before prediction:", X)
        prediction = model.predict(X)
        print("Prediction:", prediction)

        return render_template("index.html", prediction_text=f"The algorithm is {prediction}")
    
    except Exception as e:
        print("Error occurred:", str(e))
        print("Error type:", type(e).__name__)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    flask_app.run(debug=True,port=5002)
