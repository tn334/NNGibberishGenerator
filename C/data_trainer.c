// Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// global variables
#define NULL_CHAR '\0'

// structs
typedef struct TreeNode {
    char letter;
    int frequencyCount;
    struct TreeNode* children;
    struct TreeNode* next;
} TreeNode;

typedef struct FrequencyTree {
    TreeNode* root;
} FrequencyTree;

typedef struct FlatNode {
    char letter;
    int depth;
    int count;
    char parent;   
} FlatNode;

// function definitions
TreeNode* createTreeNode(char letter);
FlatNode* flattenTree(FrequencyTree* tree, char traversal_method);
FrequencyTree* createFrequencyTree();
int countNodes(TreeNode* node);
void insertWord(FrequencyTree* tree, char* word);
void printTree(TreeNode* node, char* prefix, int is_last_sibling);
void writeTreeToFile(TreeNode* node, FILE* file, char* prefix, int is_last_sibling);
void freeTreeNode(TreeNode* node);
void freeFrequencyTree(FrequencyTree* tree);
void dfs(TreeNode* node, int depth, char parent, FlatNode* output_list, int* index);
void bfs(TreeNode* rootNode, FlatNode* output_list);

// main
int main() 
{
    // open file with 370k words
    FILE* file = fopen("../Training Files/words_alpha.txt", "r");
    if (file == NULL) 
    {
        printf("Error opening file.\n");
        return 1;
    }

    // create frequency tree
    FrequencyTree* tree = createFrequencyTree();

    // read words from file
    char word[100];
    while (fscanf(file, "%s", word) != EOF) 
    {
        insertWord(tree, word);
    }
    fclose(file);

    // flatten the tree
    FlatNode* flatList = flattenTree(tree, 'd');

    // Print entire tree in terminal
    // printf("Frequency Tree:\n");
    // printTree(tree->root, "", 1);

    // Write entire tree to file
    // file = fopen("frequency_tree_output_C_Version.txt", "w");
    // if (file == NULL) {
    //     printf("Error opening file.\n");
    //     return 1;
    // }
    // writeTreeToFile(tree->root, file, "", 1);
    // fclose(file);

    // write flattened tree to file
    int totalNodes = countNodes(tree->root);
    file = fopen("../Output/flattenedTree.txt", "w");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return 1;
    }

    for (int index = 0; index < totalNodes; index++)
    {
        if (flatList[index].parent != NULL_CHAR || flatList[index].letter != NULL_CHAR)
        {
            fprintf(file, "%c, %d, %d, %c\n",
                flatList[index].letter, flatList[index].depth, flatList[index].count, flatList[index].parent);
        }
    } 
    
    // Free the entire tree and then the tree itself
    free(flatList);
    freeFrequencyTree(tree);

    return 0;
}

// function implementations
TreeNode* createTreeNode(char letter)
{
    TreeNode* newNode = (TreeNode*)malloc(sizeof(TreeNode));
    newNode->letter = letter;
    newNode->frequencyCount = 1;
    newNode->children = NULL;
    newNode->next = NULL;
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
    int letter;
    char charToInsert;
    
    TreeNode* currentNode = tree->root;
    
    for (letter = 0; word[letter] != NULL_CHAR; letter++) 
    {
        charToInsert = word[letter];
        
        TreeNode* childNode = currentNode->children;

        while (childNode != NULL) 
        {
            if (childNode->letter == charToInsert) 
            {
                childNode->frequencyCount++;
                break;
            }
            childNode = childNode->next;
        }
        
        if (childNode == NULL) {
            TreeNode* newChild = createTreeNode(charToInsert);
            newChild->next = currentNode->children;
            currentNode->children = newChild;
        }
        currentNode = currentNode->children;
    }
}

void printTree(TreeNode* node, char* prefix, int is_last_sibling) 
{
    if (node == NULL) return;

    char branch_symbol[5];
    strcpy(branch_symbol, is_last_sibling ? "L--- " : "T--- ");
    printf("%s%s%c (count: %d)\n", prefix, branch_symbol, node->letter, node->frequencyCount);

    TreeNode* child = node->children;
    while (child != NULL) 
    {
        int new_prefix_len = strlen(prefix) + 5;
        char* new_prefix = (char*)malloc(new_prefix_len * sizeof(char));
        strcpy(new_prefix, prefix);
        strcat(new_prefix, is_last_sibling ? "    " : "|   ");
        printTree(child, new_prefix, child->next == NULL);
        free(new_prefix);
        child = child->next;
    }
}

void writeTreeToFile(TreeNode* node, FILE* file, char* prefix, int is_last_sibling) 
{
    // return if empty tree
    if (node == NULL) return;

    // Write the node to the file regardless of the prefix
    fprintf(file, "%s%s%c (count: %d)\n", prefix, is_last_sibling ? "L--- " : "T--- ", node->letter, node->frequencyCount);

    // allocate memory for children prefixes
    char* childPrefix = (char*)malloc(strlen(prefix) + 5);
    strcpy(childPrefix, prefix);
    strcat(childPrefix, is_last_sibling ? "    " : "|   "); // set spacing

    // loop through children nodes
    TreeNode* child = node->children;
    while (child != NULL) 
    {
        // write child node
        writeTreeToFile(child, file, childPrefix, child->next == NULL);
        child = child->next;
    }
    free(childPrefix); // free child prefix
}

void freeTreeNode(TreeNode* node) 
{
    if (node == NULL) return;

    // Recursively free all children of the current node
    TreeNode* child = node->children;
    while (child != NULL) 
    {
        TreeNode* nextChild = child->next; // Save the next child before freeing
        freeTreeNode(child);
        child = nextChild;
    }

    // Free the current node
    free(node);
}

void freeFrequencyTree(FrequencyTree* tree) 
{
    if (tree == NULL) return;

    // Free the entire tree
    freeTreeNode(tree->root);

    // Free the tree itself
    free(tree);
}

int countNodes(TreeNode* node) 
{
    int totalNodeCount = 1;
    TreeNode* child = node->children;
    if (node == NULL) return 0;

    while (child != NULL) 
    {
        totalNodeCount += countNodes(child);
        child = child->next;
    }

    return totalNodeCount;
}

void dfs(TreeNode* node, int depth, char parent, FlatNode* output_list, int* index)
{
    if (node == NULL) return; // check empty tree

    // add node to output list
    output_list[*index].letter = node->letter;
    output_list[*index].depth = depth;
    output_list[*index].count = node->frequencyCount;
    output_list[*index].parent = parent;
    (*index)++;

    // loop through tree nodes
    TreeNode* child = node->children;
    while (child != NULL) 
    {
        dfs(child, depth + 1, node->letter, output_list, index);
        child = child->next;    
    }
}

// TODO: missing logic
void bfs(TreeNode* rootNode, FlatNode* output_list)
{
    if (rootNode == NULL) return;

    int front = 0, rear = 0, depth = 0, queueSize;
    int totalNodes = countNodes(rootNode);

    TreeNode* queue[totalNodes];
    queue[rear++] = rootNode;

    while(front != rear)
    {
        queueSize = rear - front;

        for (int i = 0; i < queueSize; i++)
        {
            TreeNode* currentNode = queue[front++];
            output_list[i].letter = currentNode->letter;
            output_list[i].depth = depth;
            output_list[i].count = currentNode->frequencyCount;
            // output_list[i].parent = ;

            TreeNode* child = currentNode->children;
            while(child != NULL)
            {
                queue[rear++] = child;
                child = child->next;
            }
        }
        depth++;
    }
}

FlatNode* flattenTree(FrequencyTree* tree, char traversal_method)
{
    int index, depth = -1;
    TreeNode* rootNode = tree->root;
    int totalNodes = countNodes(tree->root); // get total nodes in tree
    
    // malloc flat list
    FlatNode* flatList = (FlatNode*)malloc(totalNodes * sizeof(FlatNode));

    if (flatList == NULL)
    {
        printf("Memory allocation failed.");
        exit(EXIT_FAILURE);
    }

    // flatten tree with dfs
    if (traversal_method == 'd')
    {
        index = 0;
        dfs(rootNode, depth, NULL_CHAR, flatList, &index);
    }

    // flatten tree with bfs
    else if (traversal_method == 'b')
    {
        index = 0;
        bfs(rootNode, flatList);
    }

    // illegal traversal method
    else
    {
        fprintf(stderr, "Invalid traversal method. Use 'dfs' or 'bfs'.\n");
        exit(EXIT_FAILURE);
    }

    return flatList;
}
