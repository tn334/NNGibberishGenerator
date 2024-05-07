#include <iostream>
#include <fstream>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

#include <curand_kernel.h>

using namespace std;

// GPU Constants
// MODE 0 - CPU
// MODE 1 - GPU BASELINE
#define MODE 1
#define BLOCKSIZE 1024

// Global Constants
#define ALPHABET_SIZE 26
#define NODES_PER_COL 27
#define MAX_WORD_SIZE 20
#define WORDS_TO_CREATE 10000
#define COMMA ','
#define UNDERSCORE '_'
#define UNDERSCORE_INDEX 26

struct GraphNode
{
    char letter;
    float weights[NODES_PER_COL];
};

////////////////////////////////////////////////////////////////////////////////
//                                GPU FUNCTIONS                               //
////////////////////////////////////////////////////////////////////////////////
//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void warmUpGPU()
{
    printf("\nWarming up GPU for time trialing...\n");
    cudaDeviceSynchronize();
    return;
}

__global__ void createWordsBaseline(GraphNode *node_array, char *word_list);
__device__ int charToIndex(char letter);
__device__ float generateRandomFloat(int tid);
__device__ char generateRandomLetter(int tid);

////////////////////////////////////////////////////////////////////////////////
//                                CPU FUNCTIONS                               //
////////////////////////////////////////////////////////////////////////////////
int char_to_index(char letter);
void create_word(GraphNode network[][NODES_PER_COL]);
void extract_details(const string word_data, string &extracted_word,
                                                          int extracted_freq[]);
void flatten_network(GraphNode network[][NODES_PER_COL], 
                      GraphNode flattened_array[MAX_WORD_SIZE * NODES_PER_COL]);
double generate_random_float();
char generate_random_letter();
void get_char_pair(const string word, string &char_pair, const int current_char);
void initialize_network(GraphNode network[][NODES_PER_COL]);
void normalize_network_weights(GraphNode network[][NODES_PER_COL]);
void populate_network(GraphNode network[][NODES_PER_COL],
                                                    ifstream &five_letter_dict);
void print_network(GraphNode network[][NODES_PER_COL]);

int main()
{
    int word_counter, num_blocks, block_size;
    double time_start, time_end;
    ifstream five_letter_dict;

    GraphNode node_network[MAX_WORD_SIZE][NODES_PER_COL];
    GraphNode *flattened_array;
    GraphNode *dev_flattened_array;

    char *created_words;
    char *dev_created_words;

    time_start = omp_get_wtime();

    srand(time(NULL));

    warmUpGPU();

    five_letter_dict.open("../Training Files/20_letter_frequency_list_padded.txt");

    if (!five_letter_dict.is_open())
    {
        cerr << "Couldn't open file, exiting..." << endl;
        return 1;
    }

    flattened_array = (GraphNode *)
                      malloc(sizeof(GraphNode) * MAX_WORD_SIZE * NODES_PER_COL);

    created_words = (char *)
                         malloc(sizeof(char) * WORDS_TO_CREATE * MAX_WORD_SIZE);

    initialize_network(node_network);

    populate_network(node_network, five_letter_dict);

    normalize_network_weights(node_network);

    if(MODE != 0)
    {
        printf("Using GPU...\n");

        flatten_network(node_network, flattened_array);

        gpuErrchk(cudaMalloc((GraphNode**)&dev_flattened_array, 
                            sizeof(GraphNode) * MAX_WORD_SIZE * NODES_PER_COL));

        gpuErrchk(cudaMemcpy(dev_flattened_array, flattened_array, 
                              sizeof(GraphNode) * MAX_WORD_SIZE * NODES_PER_COL, 
                                                       cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((char **)&dev_created_words, 
                               sizeof(char) * WORDS_TO_CREATE * MAX_WORD_SIZE));

        num_blocks = (WORDS_TO_CREATE + BLOCKSIZE - 1) / BLOCKSIZE;;
        block_size = BLOCKSIZE;

        printf("Number of blocks created: %d\n", num_blocks);

        createWordsBaseline<<<num_blocks, block_size>>>(dev_flattened_array,
                                                             dev_created_words);

        gpuErrchk(cudaMemcpy(created_words, dev_created_words, 
       sizeof(char) * WORDS_TO_CREATE * MAX_WORD_SIZE, cudaMemcpyDeviceToHost));

        cudaFree(dev_flattened_array);

        cudaFree(dev_created_words);

        for(int i = 0; i < WORDS_TO_CREATE; i++) 
        {
            printf("%i: ", i + 1);
            for(int j = 0; j < MAX_WORD_SIZE; j++) 
            {
                printf("%c", created_words[j + i * MAX_WORD_SIZE]);
            }
            printf("\n");
        }

    }

    else
    {
        printf("Using CPU...\n");

        for(word_counter = 0; word_counter < WORDS_TO_CREATE; word_counter++)
        {
            create_word(node_network);
        }

    }

    // printf("%s\n", created_words[0]);

    // for(int i = 0; i < WORDS_TO_CREATE && MODE != 0; i++)
    // {
    //     printf("%s\n", created_words[i]);
    // }

    // print_network(node_network);

    five_letter_dict.close();

    free(flattened_array);

    free(created_words);

    time_end = omp_get_wtime();

    printf("Total time: %f\n", time_end - time_start);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//                                GPU FUNCTIONS                               //
////////////////////////////////////////////////////////////////////////////////
__global__ void createWordsBaseline(GraphNode *node_array, char *word_list)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    float cumulative_sum;
    float random_float = generateRandomFloat(tid);

    char start_letter = generateRandomLetter(tid);
    char word[MAX_WORD_SIZE];

    int letter_index = charToIndex(start_letter);
    int letter_count = 0;
    int weight_index;

    if(tid < WORDS_TO_CREATE)
    {
        word[0] = start_letter;

        for (letter_count = 1; letter_count < MAX_WORD_SIZE; letter_count++)
        {
            cumulative_sum = 0.0f;
            
            for (weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                cumulative_sum += node_array
                           [letter_index + NODES_PER_COL * letter_count].weights
                                                                 [weight_index];

                if(cumulative_sum >= random_float)
                {   
                    letter_index = weight_index;

                    word[letter_count] = node_array
                           [letter_index + NODES_PER_COL * letter_count].letter;
                    break;
                }

            }

        }

        // Copy the word to the word list
        for(int i = 0; i < MAX_WORD_SIZE; i++) 
        {
            word_list[i + tid * MAX_WORD_SIZE] = word[i]; 
        }

    }

}

__device__ int charToIndex(char letter)
{
    if(letter == UNDERSCORE)
    {
        return UNDERSCORE_INDEX;
    }

    return letter - 'a';
}

__device__ float generateRandomFloat(int tid)
{
    curandState state;
    curand_init(clock64(), tid, 0, &state);

    return curand_uniform(&state);
}

__device__ char generateRandomLetter(int tid)
{
    int random_letter;
    curandState state;
    curand_init(clock64(), tid, 0, &state);

    random_letter = curand(&state) % ALPHABET_SIZE;
    
    return 'a' + random_letter;
}

////////////////////////////////////////////////////////////////////////////////
//                                CPU FUNCTIONS                               //
////////////////////////////////////////////////////////////////////////////////
int char_to_index(char letter) 
{
    if (islower(letter)) 
    {
        return letter - 'a';
    }

    return UNDERSCORE_INDEX;
}

void create_word(GraphNode network[][NODES_PER_COL])
{ 
    generate_random_float();

    float random_float = generate_random_float();
    char start_letter = generate_random_letter();
    int letter_index = char_to_index(start_letter);
    float cumulative_sum;
    int current_row, weight_index;
    string word;

    word.clear();
    word += start_letter;

    for (current_row = 1; current_row < MAX_WORD_SIZE; current_row++)
    {
        cumulative_sum = 0.0f;

        for (weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
        {
            cumulative_sum += 
                       network[current_row][letter_index].weights[weight_index];

            if(cumulative_sum >= random_float)
            {   
                letter_index = weight_index;
                word += network[current_row][letter_index].letter;
                break;
            }

        }

    }

    cout << word << "\n";
}

void extract_details(const string word_data, string &extracted_word,
                                                           int extracted_freq[])
{
    int freq_index = 0;
    size_t index = 0;
    size_t digit_index;
    string current_freq;

    extracted_word.clear();

    for (index = 0; index < word_data.length(); index++)
    {
        if (isalpha(word_data[index]) || word_data[index] == UNDERSCORE)
        {
            extracted_word += word_data[index];
        }

        else if (isdigit(word_data[index]))
        {
            digit_index = index;

            while (word_data[digit_index] != COMMA &&
                                               digit_index < word_data.length())                                        
            {
                digit_index++;
            }

            current_freq = word_data.substr(index, digit_index - index);

            extracted_freq[freq_index] = stoi(current_freq);

            freq_index++;

            index = digit_index;
        }

    }

}

void flatten_network(GraphNode network[][NODES_PER_COL], 
                       GraphNode flattened_array[MAX_WORD_SIZE * NODES_PER_COL])
{   

    int current_col, current_node;

    for (current_col = 0; current_col < MAX_WORD_SIZE; current_col++)
    {

        for (current_node = 0; current_node < NODES_PER_COL; current_node++)
        {
            flattened_array[current_node + NODES_PER_COL * current_col] = 
                                             network[current_col][current_node];
        }

    }

}

double generate_random_float()
{
    return (float)(rand()) / (float)(RAND_MAX);
}

char generate_random_letter()
{
    return 'a' + (rand() % ALPHABET_SIZE);
}

void get_char_pair(const string word, string &char_pair, const int current_char)
{
    char_pair.clear();
    char_pair += word[current_char];
    char_pair += word[current_char + 1];
}

void initialize_network(GraphNode network[][NODES_PER_COL])
{
    int current_row, current_node, weight_index;
    char letter;
    GraphNode new_node;

    for (current_row = 0; current_row < MAX_WORD_SIZE; current_row++)
    {
        letter = 'a';

        for (current_node = 0; current_node < NODES_PER_COL; current_node++)
        {
            for(weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                new_node.weights[weight_index] = 0;
            }

            if(current_node < ALPHABET_SIZE)
            {
                new_node.letter = letter;
            }

            else
            {
                new_node.letter = UNDERSCORE;
            }

            network[current_row][current_node] = new_node;

            letter = (char)((int)letter + 1);
        }

    }

}

void normalize_network_weights(GraphNode network[][NODES_PER_COL])
{
    int current_row, current_node, weight_index;
    float total_weight_sum;

    for(current_row = 0; current_row < MAX_WORD_SIZE; current_row++)
    {
        for(current_node = 0; current_node < NODES_PER_COL; current_node++)
        {
            total_weight_sum = 0;

            for(weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                total_weight_sum += 
                       network[current_row][current_node].weights[weight_index];
            }

            for(weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                if(total_weight_sum != 0)
                {
                    network[current_row][current_node].weights[weight_index] 
                                                            /= total_weight_sum;
                }

            }

        }

    }

}

void populate_network(GraphNode network[][NODES_PER_COL],
                                                     ifstream &five_letter_dict)
{
    string word;
    string word_data;
    string char_pair;
    size_t current_char;
    int frequencies[MAX_WORD_SIZE];
    int current_row, current_node;
    int weight_index;

    while (getline(five_letter_dict, word_data))
    {
        current_row = 0;  // row
        current_node = 0; // col
        weight_index = 0;

        extract_details(word_data, word, frequencies);

        for (current_char = 0; current_char < word.length() - 1; current_char++)
        {
            get_char_pair(word, char_pair, current_char);

            current_node = char_to_index(char_pair[0]);

            weight_index = char_to_index(char_pair[1]);

            network[current_row][current_node].weights[weight_index]++;

            current_row++;
        }

        memset(frequencies, 0, sizeof(frequencies));
    }

}

void print_network(GraphNode network[][NODES_PER_COL])
{
    int current_row, current_node, weight_index;

    for (current_row = 0; current_row < MAX_WORD_SIZE; current_row++)
    {
        for (current_node = 0; current_node < NODES_PER_COL; current_node++)
        {
            printf("Network[%d][%d]: letter = %c, weights:\n", current_row, 
                    current_node, network[current_row][current_node].letter);

            for(weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                printf("   Weight for %c: %f\n", 
                                      network[current_row][weight_index].letter, 
                      network[current_row][current_node].weights[weight_index]);
            }

        }

    }

}