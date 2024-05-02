// Header Files
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// Global Constants
#define NUMBER_OF_LETTERS 26
#define MAX_WORD_SIZE 5
#define DICT_WORD_COUNT 25689
#define COMMA ','

// Function Declarations
// void updateFinalArray(int final_array[][MAX_WORD_SIZE][NUMBER_OF_LETTERS], const int *digit_array);

// Main Function
int main()
{
    // Initialize Variables
    FILE *five_letter_dict;
    int current_array_idx = 0;
    char current_char;

    // Initialize Arrays
    int final_array[DICT_WORD_COUNT][MAX_WORD_SIZE][NUMBER_OF_LETTERS];
    int current_num_array[NUMBER_OF_LETTERS];

    // Open file with given path on reading mode
    five_letter_dict = fopen("../Output/numerical_5_letter_dict_simplified.txt", "r");

    // Check if the file opening succeeded
    if(five_letter_dict == NULL)
    {
        // Print error message
        printf("Couldn't open file, exiting...\n");

        // Exit program with failure
        return -1;
    }

    // Prime file read by getting the first character (first digit)
    current_char = fgetc(five_letter_dict);

    // // Loop through all contents of file
    // while(current_char != '\n')
    // {
    //     // Check if the current file character is a number
    //     if(isdigit(current_char))
    //     {
    //         // Convert the current character to its former digit and set it in 
    //         // array
    //         current_num_array[current_array_idx] = current_char - '0';

    //         // Increment current number array index
    //         current_array_idx++;
    //     }

    //     // Otherwise, check if the current file character is a comma                                                                                                                                    
    //     else if(current_char == COMMA)
    //     {
    //         printf("\nFound comma, pushing values of digit to final array\n");

    //         current_array_idx = 0;

    //         // Set the current number array at the final array to be exported
    //         updateFinalArray(final_array, current_num_array);
    //     }

    //     // Otherwise, assume file character is a new line character
    //     // else
    //     // {
    //     // }

    //     // Read the next character from the file
    //     current_char = fgetc(five_letter_dict);
    // }

    // Close opened file
    fclose(five_letter_dict);

    // Return success
    return 0;
}

// Update Final Array Function
// void updateFinalArray(int final_array[][MAX_WORD_SIZE][NUMBER_OF_LETTERS], const int *digit_array)
// {

// }

// printf("current_num_array[%d] = %d\n", current_array_idx, current_num_array[current_array_idx]);