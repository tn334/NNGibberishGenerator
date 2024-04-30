#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

using namespace std;

#define NODES_PER_COL 26
#define MAX_WORD_SIZE 5

struct GraphNode
{
    char letter;
    float weights[NODES_PER_COL];
}
;

int main()
{
    int current_letter, current_node;
    char letter;
    GraphNode node_network[MAX_WORD_SIZE][NODES_PER_COL];
    GraphNode new_node;

    for(current_letter = 0; current_letter < MAX_WORD_SIZE; current_letter++)
    {
        letter = 'a';

        for(current_node = 0; current_node < NODES_PER_COL; current_node++)
        {
            new_node.letter = letter;
            new_node.weights[current_node] = 0;

            node_network[current_letter][current_node] = new_node; 

            letter = (char)((int)letter + 1);
        }

    }
    
    for(current_letter = 0; current_letter < MAX_WORD_SIZE; current_letter++)
    {
        for(current_node = 0; current_node < NODES_PER_COL; current_node++)
        {
            printf("Network[%d][%d]: letter = %c\n", current_letter, current_node, node_network[current_letter][current_node].letter);
        }

    }

    return 0;
}