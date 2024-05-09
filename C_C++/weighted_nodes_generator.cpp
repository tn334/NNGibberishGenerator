#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <ctime>

using namespace std;

#define NODES_PER_COL 26
#define MAX_WORD_SIZE 5
#define COMMA ','

struct GraphNode
{
    char letter;
    float weights[NODES_PER_COL];
    unsigned int count;
};

int char_to_index(char letter);
void create_word(GraphNode network[][NODES_PER_COL]);
void extract_details(const string word_data, string &extracted_word,
                                                          int extracted_freq[]);
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
    srand(time(NULL));
    ifstream five_letter_dict;
    GraphNode node_network[MAX_WORD_SIZE][NODES_PER_COL];

    five_letter_dict.open("5_letter_frequency_list.txt");

    if (!five_letter_dict.is_open())
    {
        cerr << "Couldn't open file, exiting..." << endl;
        return 1;
    }

    initialize_network(node_network);

    populate_network(node_network, five_letter_dict);

    normalize_network_weights(node_network);

    create_word(node_network);

    // print_network(node_network);

    five_letter_dict.close();

    return 0;
}

int char_to_index(char letter) 
{
    if (islower(letter)) 
    {
        return letter - 'a';
    }

    return letter - 'A';
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

    for (current_row = 0; current_row < MAX_WORD_SIZE; current_row++)
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

    cout << word << "\n\n";
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
        if (isalpha(word_data[index]))
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
            new_node.letter = letter;
            new_node.count = 0;

            for(weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                new_node.weights[weight_index] = 0;
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
                total_weight_sum += network[current_row][current_node].weights[weight_index];
            }

            for(weight_index = 0; weight_index < NODES_PER_COL; weight_index++)
            {
                if(total_weight_sum != 0)
                {
                    network[current_row][current_node].weights[weight_index] /= total_weight_sum;
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
        current_row = 0; // row
        current_node = 0;   // col
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
                printf("   Weight #%d: %f\n", weight_index, 
                   network[current_row][current_node].weights[weight_index]);
            }

        }

    }

}