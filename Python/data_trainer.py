from collections import deque
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

def writeTraversalToFile(result_list):
    #write entire result_list to file
    with open("/../Output/5_letter_frequency_list_padded.txt", "w") as output_file:
      for item in result_list:
            combined_values = []
            for i in range(0, len(item), 2):
                combined_values.append(str(item[i]) + str(item[i + 1]))
            output_file.write(",".join(combined_values) + "\n")

def traverse_tree(node, result_list, length, current_list = None):

  # set local copy of current_list
  if current_list is None:
      current_list = []
  
  # Append current node's letter and freequency to the current list
  current_list.append(node.letter)
  current_list.append(node.frequencyCount)

  # Check if the node is an end node
  if len(node.children) == 0:
    word_list = []
    word_list = current_list[2:]
    
    while length < 5:
      word_list.append('_')
      word_list.append(1)
      length+=1

    # Append the list to the result list
    result_list.append(word_list)

    return

  length+=1
  # Iterate through the children of the node
  for child_node in node.children.values():
      
      # Recursively traverse the child node
      traverse_tree(child_node, result_list, length, current_list)
      
      # Remove the current node's letter and frequency from the current list
      for i in range(2):
        current_list.pop()

def writeTraversalToFileCommaSep(result_list):
    #write entire result_list to file
    with open("/../Output/5_letter_frequency_list_padded_comma_sep.txt", "w") as output_file:
      for item in result_list:
        output_file.write(','.join(map(str, item))+"\n")

class FlatNode():
    def __init__(self, label, depth, count, parent):
        self.label = label
        self.depth = depth
        self.count = count
        self.parent = parent

def flattenTree(root_node, output_list, traversal_method='dfs'):
    if traversal_method == 'dfs':
        def dfs(node, depth, parent):
            if not node.children:
                return

            for child in node.children.values():
                output_list.append(FlatNode(child.letter, depth, child.frequencyCount, parent))
                dfs(child, depth + 1, child.letter)
        dfs(root_node, 0, None)

    elif traversal_method == 'bfs':
        queue = deque([(root_node, 0, None)])
        while queue:
            node, depth, parent = queue.popleft()
            output_list.append(FlatNode(node.letter, depth, node.frequencyCount, parent))
            for child in node.children.values():
                queue.append((child, depth + 1, node.letter))
                
    else:
        raise ValueError("Invalid traversal method. Use 'dfs' or 'bfs'.")

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
    with open('../Output/5_letter_dict.txt', 'r') as file:
        words = file.readlines()

    tree = FrequencyTree()

    for word in words:
        word = word.strip()
        tree.insertWord(word)

    # output_list_dfs = flattenTree(tree.root, [], 'dfs')
    # output_list_bfs = flattenTree(tree.root, [], 'bfs')

    # print flattened tree using dfs
    # print("DFS Output list")
    # for flat_node in output_list_dfs:
    #     print(f"Label: {flat_node.label}, Depth: {flat_node.depth}, Count: {flat_node.count}, Parent: {flat_node.parent}")

    # print flattened tree using bfs
    # print("BFS Output list")
    # for flat_node in output_list_bfs:
        # print(f"Label: {flat_node.label}, Depth: {flat_node.depth}, Count: {flat_node.count}, Parent: {flat_node.parent}")

    # print entire tree in terminal    
    # tree.printTree()

    # write entire tree to file
    # writeTreeToFile(tree, 'frequency_tree_output_Python_Version.txt')

    # Create an empty list to store the result
    result_list = []
    length = 0

    # Traverse the frequency tree
    traverse_tree(tree.root, [], length, result_list)

    # Print the result list
    print(result_list)

    # create 5 letter dictionary
    # with open("../Training Files/words_alpha.txt", "r") as input_file:
    #     lines = input_file.readlines()

    # with open("../Output/5_letter_dict.txt", "w") as file_out:
    #     for line in lines:
    #         words = line.split()

    #         for word in words:
    #             if len(word) <= 5:
    #                 file_out.write(word + "\n")
    #write entire result_list to file
    #writeTraversalToFile(result_list)

    writeTraversalToFileCommaSep(result_list)

if __name__ == "__main__":
    main()





