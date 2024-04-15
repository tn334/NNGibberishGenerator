// Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// global variables
#define NULL_CHAR '\0'

// structs
typedef struct TreeNode {
    char letter;
    int frequencyCount;
    struct TreeNode** children;
} TreeNode;

typedef struct FrequencyTree {
    TreeNode* root;
} FrequencyTree;

// function definitions
TreeNode* createTreeNode(char letter);
FrequencyTree* createFrequencyTree();
void insertWord(FrequencyTree* tree, char* word);
void printTree(TreeNode* node, char* prefix, int is_last_sibling);
void writeTreeToFile(TreeNode* node, FILE* file, char* prefix, int is_last_sibling);
void freeFrequencyTree(TreeNode* node);

// main
int main() {
    // open file with 370k words
    FILE* file = fopen("words_alpha.txt", "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    // create frequency tree
    FrequencyTree* tree = createFrequencyTree();

    // read words from file
    char word[100];
    while (fscanf(file, "%s", word) != EOF) {
        insertWord(tree, word);
    }
    fclose(file);

    // Print entire tree in terminal
    // printf("Frequency Tree:\n");
    // printTree(tree->root, "", 1);

    // Write entire tree to file
    file = fopen("frequency_tree_output_C_Version.txt", "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }
    writeTreeToFile(tree->root, file, "", 1);
    fclose(file);

    // Free the FrequencyTree
    freeFrequencyTree(tree->root);
    free(tree);

    return 0;
}

// function implementations
TreeNode* createTreeNode(char letter)
{
    TreeNode* newNode = (TreeNode*) malloc(sizeof(TreeNode));
    newNode->letter = letter;
    newNode->frequencyCount = 0;
    newNode->children = (TreeNode**)malloc(26 * sizeof(TreeNode*));
    for (int index = 0; index < 26; index++)
    {
        newNode->children[index] = NULL;
    }
    return newNode;
}

FrequencyTree* createFrequencyTree()
{
    FrequencyTree* tree = (FrequencyTree*) malloc(sizeof(FrequencyTree));
    tree->root = createTreeNode(NULL_CHAR);
    return tree;
}

void insertWord(FrequencyTree* tree, char* word)
{
    int letter, charIndex;
    TreeNode* currentNode = tree->root;

    for (letter = 0; word[letter] != NULL_CHAR; letter++)
    {
        charIndex = word[letter] - 'a';
        if (currentNode->children[charIndex] == NULL)
        {
            currentNode->children[charIndex] = createTreeNode(word[letter]);
        }
        currentNode = currentNode->children[charIndex];
        currentNode->frequencyCount++;
    }
}

void printTree(TreeNode* node, char* prefix, int is_last_sibling) 
{
    // return if empty tree
    if (node == NULL) return;

    // check for prefix not empty
    if (strlen(prefix) > 0) 
    {
        printf("%s", prefix);
        printf(is_last_sibling ? "└── " : "├── "); // same as if true do_this else do_that
        printf("%c (count: %d)\n", node->letter, node->frequencyCount);
    }

    // allocate memory for children prefixes
    char* childPrefix = (char*)malloc(strlen(prefix) + 5);
    strcpy(childPrefix, prefix);

    // loop through children nodes
    for (int i = 0; i < 26; i++) 
    {
        // check node is not empty
        if (node->children[i] != NULL) 
        {
            strcat(childPrefix, is_last_sibling ? "    " : "│   "); // set spacing
            // print child node
            printTree(node->children[i], childPrefix, (node->children[i+1] == NULL) ? 1 : 0);
            childPrefix[strlen(prefix)] = NULL_CHAR; // reset node
        }
    }
    free(childPrefix); // free child prefix
}

void writeTreeToFile(TreeNode* node, FILE* file, char* prefix, int is_last_sibling) 
{
    // return if empty tree
    if (node == NULL) return;

    // check for prefix not empty
    if (strlen(prefix) > 0) 
    {
        fprintf(file, "%s", prefix);
        fprintf(file, is_last_sibling ? "└── " : "├── "); // same as if true do_this else do_that
        fprintf(file, "%c (count: %d)\n", node->letter, node->frequencyCount);
    }

    // allocate memory for children prefixes
    char* childPrefix = (char*)malloc(strlen(prefix) + 5);
    strcpy(childPrefix, prefix);

    // loop through children nodes
    for (int i = 0; i < 26; i++) 
    {
        // check node is not empty
        if (node->children[i] != NULL) 
        {
            strcat(childPrefix, is_last_sibling ? "    " : "│   "); // set spacing
            // write child node
            writeTreeToFile(node->children[i], file, childPrefix, (node->children[i+1] == NULL) ? 1 : 0);
            childPrefix[strlen(prefix)] = NULL_CHAR; // reset node
        }
    }
    free(childPrefix); // free child prefix
}

void freeFrequencyTree(TreeNode* node) 
{
    if (node == NULL) return;
    for (int i = 0; i < 26; i++) 
    {
        freeFrequencyTree(node->children[i]);
    }
    free(node->children);
    free(node);
}
