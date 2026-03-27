#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <atomic>
#include <future>
#include <execution>
using namespace std;


const int ROWS = 1000;
const int COLS = 10000;
const int WORDS_PER_ROW = (COLS + 63) / 64;  
const int WORDS_PER_COL = (ROWS + 63) / 64;  


#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__popcnt64)
inline int popcount64(uint64_t x) {
    return (int)__popcnt64(x);
}
#else
inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}
#endif


struct RowBits {
    uint64_t words[WORDS_PER_ROW];

    RowBits() {
        memset(words, 0, sizeof(words));
    }

    void set_bit(int col) {
        int word_idx = col / 64;
        int bit_idx = col % 64;
        words[word_idx] |= (1ULL << bit_idx);
    }
};


struct ColBits {
    uint64_t words[WORDS_PER_COL];

    ColBits() {
        memset(words, 0, sizeof(words));
    }

    void set_bit(int row) {
        int word_idx = row / 64;
        int bit_idx = row % 64;
        words[word_idx] |= (1ULL << bit_idx);
    }
};


struct VectorBits {
    uint64_t words[WORDS_PER_ROW];
    int length;

    VectorBits(int len = COLS) : length(len) {
        memset(words, 0, sizeof(words));
    }

    void set_bit(int pos) {
        int word_idx = pos / 64;
        int bit_idx = pos % 64;
        words[word_idx] |= (1ULL << bit_idx);
    }

    bool get_bit(int pos) const {
        int word_idx = pos / 64;
        int bit_idx = pos % 64;
        return (words[word_idx] >> bit_idx) & 1;
    }

    void flip_bit(int pos) {
        int word_idx = pos / 64;
        int bit_idx = pos % 64;
        words[word_idx] ^= (1ULL << bit_idx);
    }

    bool is_zero() const {
        for (int i = 0; i < WORDS_PER_ROW; i++) {
            if (words[i] != 0) return false;
        }
        return true;
    }

    VectorBits& operator=(const VectorBits& other) {
        if (this != &other) {
            length = other.length;
            memcpy(words, other.words, sizeof(words));
        }
        return *this;
    }
};


struct SyndromeBits {
    uint64_t words[WORDS_PER_COL];

    SyndromeBits() {
        memset(words, 0, sizeof(words));
    }

    void set_bit(int row) {
        int word_idx = row / 64;
        int bit_idx = row % 64;
        words[word_idx] |= (1ULL << bit_idx);
    }

    bool is_zero() const {
        for (int i = 0; i < WORDS_PER_COL; i++) {
            if (words[i] != 0) return false;
        }
        return true;
    }
};

struct H {
    vector<RowBits> rows;
    vector<ColBits> cols;
    vector<int> col_weights;

    H(int rows_count, int cols_count) {
        rows.resize(rows_count);
        cols.resize(cols_count);
        col_weights.resize(cols_count, 0);
    }

    void set_bit(int row, int col) {
        rows[row].set_bit(col);
        cols[col].set_bit(row);
        col_weights[col]++;
    }

    int get_col_weight(int col) const {
        return col_weights[col];
    }
};


void init_matrix(H& h, int ones_per_col) {
    const int rows = ROWS;
    const int cols = COLS;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, rows - 1);

    for (int col = 0; col < cols; col++) {
        vector<bool> used(rows, false);

        for (int i = 0; i < ones_per_col; i++) {
            int pos;
            do {
                pos = dist(gen);
            } while (used[pos]);
            used[pos] = true;
            h.set_bit(pos, col);
        }
    }
}

int count_bits(uint64_t x) {
    return popcount64(x);
}

SyndromeBits compute_syndrome_parallel(const VectorBits& y, const H& h, int num_threads) {
    const int rows = (int)h.rows.size();
    SyndromeBits syndrome;

    vector<future<vector<int>>> futures;
    int rows_per_thread = rows / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? rows : (t + 1) * rows_per_thread;

        futures.push_back(async(launch::async, [&, start_row, end_row]() {
            vector<int> local_syndrome_rows;
            for (int i = start_row; i < end_row; i++) {
                uint64_t parity = 0;
                for (int w = 0; w < WORDS_PER_ROW; w++) {
                    parity ^= (h.rows[i].words[w] & y.words[w]);
                }

                int bit_count = count_bits(parity);
                if (bit_count % 2 == 1) {
                    local_syndrome_rows.push_back(i);
                }
            }
            return local_syndrome_rows;
            }));
    }

    
    for (auto& f : futures) {
        vector<int> rows_with_error = f.get();
        for (int row : rows_with_error) {
            syndrome.set_bit(row);
        }
    }

    return syndrome;
}


vector<int> compute_f_parallel(const SyndromeBits& syndrome, const H& h, int num_threads) {
    const int cols = COLS;
    vector<int> f(cols, 0);


    vector<future<void>> futures;
    int cols_per_thread = cols / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start_col = t * cols_per_thread;
        int end_col = (t == num_threads - 1) ? cols : (t + 1) * cols_per_thread;

        futures.push_back(async(launch::async, [&, start_col, end_col]() {
            for (int j = start_col; j < end_col; j++) {
                int sum = 0;
                for (int w = 0; w < WORDS_PER_COL; w++) {
                    sum += count_bits(h.cols[j].words[w] & syndrome.words[w]);
                }
                f[j] = sum;
            }
            }));
    }

    for (auto& f : futures) {
        f.wait();
    }

    return f;
}


bool flip_bits_parallel(VectorBits& y, const vector<int>& f, const H& h, int num_threads) {
    const int cols = COLS;
    atomic<bool> bits_flipped{ false };


    vector<future<void>> futures;
    int cols_per_thread = cols / num_threads;

    float threshold = h.get_col_weight(0) / 2;

    for (int t = 0; t < num_threads; t++) {
        int start_col = t * cols_per_thread;
        int end_col = (t == num_threads - 1) ? cols : (t + 1) * cols_per_thread;
              

        futures.push_back(async(launch::async, [&, start_col, end_col]() {
            bool local_flipped = false;
            for (int j = start_col; j < end_col; j++) {
                
                if ((float)f[j] > threshold) {
                    y.flip_bit(j);
                    local_flipped = true;
                }
            }
            if (local_flipped) {
                bits_flipped = true;
            }
            }));
    }

    for (auto& f : futures) {
        f.wait();
    }

    return bits_flipped;
}


bool decode_parallel_parallel(VectorBits& received, const H& h, int num_threads) {
    const int max_iterations = 100;

    VectorBits y = received;

    for (int iter = 0; iter < max_iterations; iter++) {
        
        SyndromeBits syndrome = compute_syndrome_parallel(y, h, num_threads);


        if (syndrome.is_zero()) {
            
            return true;
        }

        
        vector<int> f = compute_f_parallel(syndrome, h, num_threads);


        bool bits_flipped = flip_bits_parallel(y, f, h, num_threads);

        if (!bits_flipped) {
            return false;
        }
    }

    return false;
}

bool decode_sequential_parallel(VectorBits& received, const H& h, int num_threads) {
    const int max_iterations = 100;
    const int cols = COLS;

    VectorBits y = received;

    for (int iter = 0; iter < max_iterations; iter++) {
        
        SyndromeBits syndrome = compute_syndrome_parallel(y, h, num_threads);

        if (syndrome.is_zero()) {
            
            return true;
        }


        vector<int> f = compute_f_parallel(syndrome, h, num_threads);


        int flip_pos = -1;
        float threshold = h.get_col_weight(0) / 2;
        for (int j = 0; j < cols; j++) {
            
            if ((float)f[j] > threshold) {
                flip_pos = j;
                break;
            }
        }

        if (flip_pos == -1) {
            return false;
        }

        y.flip_bit(flip_pos);
    }

    return false;
}


VectorBits generate_vector_bernoulli(mt19937& gen, double p) {
    VectorBits vec(COLS);
    bernoulli_distribution dist(p);

    for (int i = 0; i < COLS; i++) {
        if (dist(gen)) {
            vec.set_bit(i);
        }
    }

    return vec;
}


struct DecodingStats {
    long long total_vectors = 0;
    long long undecoded_vectors = 0;

    double get_undecoded_ratio() const {
        return total_vectors > 0 ? static_cast<double>(undecoded_vectors) / total_vectors : 0.0;
    }
};

struct SimpleStats {
    int flip_count;
    SimpleStats() : flip_count(0) {}
};

bool my_decoding(VectorBits& received, const H& h, int num_threads) {
    const int max_iterations = 100;
    const int cols = COLS;

    VectorBits y = received;
    vector<SimpleStats> bit_stats(cols);

    for (int iter = 0; iter < max_iterations; iter++) {
        
        SyndromeBits syndrome = compute_syndrome_parallel(y, h, num_threads);

        if (syndrome.is_zero()) {
            //received = y;
            return true;
        }

       
        vector<int> f = compute_f_parallel(syndrome, h, num_threads);

        
        vector<pair<int, int>> candidates; // (flip_count, position)

        float threshold = h.get_col_weight(0) / 2;
        for (int j = 0; j < cols; j++) {
            
            if ((float)f[j] > threshold) {
                
                candidates.push_back({ bit_stats[j].flip_count, j });
            }
        }

        if (candidates.empty()) {
            return false;
        }

        
        sort(candidates.begin(), candidates.end());

        
        int bits_to_flip = min(10, (int)candidates.size());

        for (int i = 0; i < bits_to_flip; i++) {
            int bit_pos = candidates[i].second;
            y.flip_bit(bit_pos);
            bit_stats[bit_pos].flip_count++;
        }
    }

    return false;
}

int main() {
    cout << "Initializing parity check matrices..." << endl;
    

    H h1(ROWS, COLS);
    H h2(ROWS, COLS);

    auto start = chrono::high_resolution_clock::now();
    init_matrix(h1, 2);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "h1 initialized (2 ones/col) in " << duration.count() << " seconds" << endl;
    cout.flush();

    start = chrono::high_resolution_clock::now();
    init_matrix(h2, 3);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "h2 initialized (3 ones/col) in " << duration.count() << " seconds" << endl;
    cout.flush();

    random_device rd;
    mt19937 gen(rd());

    const long long num_vectors = 10;  


    unsigned int num_threads = thread::hardware_concurrency();
    
    cout << "\nUsing " << num_threads << " threads for parallel computations" << endl;
    
    vector<double> p_values;
    double start_p = 0.0;
    double end_p = 0.003;
    double step = 0.0005;

    for (double p = start_p; p <= end_p + 1e-10; p += step) {
        p_values.push_back(p);
    }
    cout << "=== Testing with h1 (2 ones per column) ===" << endl;
    for (double p : p_values) {
        cout << "\nTesting p = " << p << endl;
        cout.flush();

        DecodingStats stats_posl_h1;
        DecodingStats stats_parall_h1;
        DecodingStats stats_my_h1;
        auto start_time = chrono::high_resolution_clock::now();

        for (long long i = 0; i < num_vectors; i++) {
            VectorBits received = generate_vector_bernoulli(gen, p);
            stats_posl_h1.total_vectors++;
            stats_parall_h1.total_vectors++;
            stats_my_h1.total_vectors++;
            bool decoded = my_decoding(received, h1, num_threads);
            if (!decoded) {
                stats_my_h1.undecoded_vectors++;
            }
            
            decoded = decode_sequential_parallel(received, h1, num_threads);
            if (!decoded) {
                stats_posl_h1.undecoded_vectors++;
            }
            decoded = decode_parallel_parallel(received, h1, num_threads);
            if (!decoded) {
                stats_parall_h1.undecoded_vectors++;
            }
            
        }
        cout << "Statistics:\n";
        cout << "Sequential: incorrectly decoded frames/Total frames: " << stats_posl_h1.get_undecoded_ratio()<< endl;
        cout << "Parallel: incorrectly decoded frames/Total frames: " << stats_parall_h1.get_undecoded_ratio() << endl;
        cout << "My: incorrectly decoded frames/Total frames: " << stats_my_h1.get_undecoded_ratio() << endl;

        auto end_time = chrono::high_resolution_clock::now();
        auto duration_sec = chrono::duration_cast<chrono::seconds>(end_time - start_time);

        cout << "  Time: " << duration_sec.count() << " seconds" << endl;
              
    }

    cout << "=== Testing with h2 (3 ones per column) ===" << endl;
    for (double p : p_values) {
        cout << "\nTesting p = " << p << endl;
        cout.flush();

        DecodingStats stats_posl_h2;
        DecodingStats stats_parall_h2;
        DecodingStats stats_my_h2;
        auto start_time = chrono::high_resolution_clock::now();

        for (long long i = 0; i < num_vectors; i++) {
            VectorBits received = generate_vector_bernoulli(gen, p);
            stats_posl_h2.total_vectors++;
            stats_parall_h2.total_vectors++;
            stats_my_h2.total_vectors++;
            bool decoded = my_decoding(received, h1, num_threads);
            if (!decoded) {
                stats_my_h2.undecoded_vectors++;
            }

            decoded = decode_sequential_parallel(received, h2, num_threads);//decode_sequential(received, h1);
            if (!decoded) {
                stats_posl_h2.undecoded_vectors++;
            }
            decoded = decode_parallel_parallel(received, h2, num_threads);
            if (!decoded) {
                stats_parall_h2.undecoded_vectors++;
            }
            


        }
        cout << "Statistics:\n";
        cout << "Sequential: incorrectly decoded frames/Total frames: " << stats_posl_h2.get_undecoded_ratio() << endl;
        cout << "Parallel: incorrectly decoded frames/Total frames: " << stats_parall_h2.get_undecoded_ratio() << endl;
        cout << "My: incorrectly decoded frames/Total frames: " << stats_my_h2.get_undecoded_ratio() << endl;

        auto end_time = chrono::high_resolution_clock::now();
        auto duration_sec = chrono::duration_cast<chrono::seconds>(end_time - start_time);

        cout << "  Time: " << duration_sec.count() << " seconds" << endl;


       
    }

    
    
    cout << "\nPress Enter to exit..." << endl;
    cin.get();
   
    return 0;
}
