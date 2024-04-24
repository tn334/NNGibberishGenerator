
# classes to handle creation of frequency tree
class TreeNode:
    def __init__(self) -> None:
        self.letter = ""
        self.frequencyCount = 0
        self.children = dict()

class FrequencyTree:
    def __init__(self) -> None:
        self.root = TreeNode() # set root of tree
    
    def insertWord(self, word):
        currentNode = self.root # wkg pointer for root

        # loop through letters in word
        for char in word:

            # check if letter is not a child of current node
            if char not in currentNode.children:
                currentNode.children[char] = TreeNode() # add child node
                currentNode.children[char].letter = char # store letter in node
            
            # move to next letter (node) and increase frequency count of letter at node
            currentNode = currentNode.children[char]
            currentNode.frequencyCount += 1
        

    def printTree(self, node=None, prefix="", is_last_sibling=True):
        if node is None:
            node = self.root

        else:  # Printing nodes other than the root
            branch_symbol = "└── " if is_last_sibling else "├── "
            print(f"{prefix}{branch_symbol}{node.letter} (count: {node.frequencyCount})")
            prefix += "    " if is_last_sibling else "│   " # adjust output based on next nodes siblings
        
        children = list(node.children.values())
        for currentIteration, child in enumerate(children):
            # Pass the updated prefix and is_last flag based on whether the child is the last in its siblings
            self.printTree(child, prefix, currentIteration == len(children) - 1)

class FlatNode():
    def __init__(self, label, depth, count, parent):
        self.label = label
        self.depth = depth
        self.count = count
        self.parent = parent

def flattenTree(root_node, output_list):
    def dfs(node, depth, parent):
        if not node.children:
            return

        for child in node.children.values():
            output_list.append(FlatNode(child.letter, depth, child.frequencyCount, parent))
            dfs(child, depth + 1, child.letter)
    
    dfs(root_node, 0, None)
    return output_list


def writeTreeToFile(tree, filename):
    # Open the file in write mode with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as file:
        def printTreeToFile(node=None, prefix="", is_last_sibling=True):
            if node is None:
                node = tree.root
            else: # Printing nodes other than the root
                branch_symbol = "└── " if is_last_sibling else "├── "
                file.write(f"{prefix}{branch_symbol}{node.letter} (count: {node.frequencyCount})\n")
                prefix += "    " if is_last_sibling else "│   " # adjust output based on next nodes siblings
            
            children = list(node.children.values())
            for currentIteration, child in enumerate(children):
                # Pass the updated prefix and is_last flag based on whether the child is the last in its siblings
                printTreeToFile(child, prefix, currentIteration == len(children) - 1)
        
        # Start the recursive function call
        printTreeToFile()

def main():
    # read all words in txt file
    with open('../Training Files/words_alpha.txt', 'r') as file:
        words = file.readlines()

    tree = FrequencyTree()

    for word in words:
        word = word.strip()
        tree.insertWord(word)

    outputList = []
    flattenTree(tree.root, outputList)

    for flat_node in outputList:
        print(f"Label: {flat_node.label}, Depth: {flat_node.depth}, Count: {flat_node.count}, Parent: {flat_node.parent}")

    # print entire tree in terminal    
    # tree.printTree()

    # write entire tree to file
    # writeTreeToFile(tree, 'frequency_tree_output_Python_Version.txt')

if __name__ == "__main__":
    main()


############################   OLD CODE THAT CAN BE USEFUL   ###################################   
#frequencyDict = { letter: {} for letter in 'abcdefghijklmnopqrstuvwxyz'}

# # loop through words
# for word in words:
#     word = word.strip()

#     # compare position to letter and count positions
#     for position, letter in enumerate(word):
#         if position not in frequencyDict[letter]:
#             frequencyDict[letter][position] = 1
#         else:
#             frequencyDict[letter][position] += 1

# # write freq dict contents to file
# with open('letter_frequency.txt', 'w') as file:
#     for letter, positions in frequencyDict.items():
#         file.write(f"{letter}: {positions}\n\n")




